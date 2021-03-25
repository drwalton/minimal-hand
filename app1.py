import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import pyvirtualcam
import zmq
from scipy import ndimage
from zmq.eventloop import ioloop, zmqstream

from utils import imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa: E402
import logging  # noqa: E402

tf.get_logger().setLevel(logging.ERROR)
from wrappers import ModelPipeline  # noqa: E402


class WebcamCapture(cv2.VideoCapture):
    def __init__(self):
        super().__init__(index=0)

    def read(self, image=None):
        retval, image = super().read(image)
        assert retval, 'No image'
        return image

    def read_rgb(self):
        return cv2.cvtColor(self.read(), cv2.COLOR_BGR2RGB)


def to_vec(pos):
    return dict(zip(['x', 'y', 'z'], pos))


def filter_nan_inf(x):
    return x[np.isfinite(x)]


class FPS:
    def __init__(self, prefix):
        self.prefix = prefix
        self.tic = None
        self.fps = None
        self.upd_rate = 0.001

    def __call__(self):
        if self.tic is None:
            self.tic = time.time()
            return

        toc = time.time()
        dt = toc - self.tic
        self.tic = toc
        if dt > 0:
            if self.fps is None:
                self.fps = 1 / dt
            else:
                self.fps = self.fps + self.upd_rate * (1 / dt - self.fps)
        print(self.prefix, self.fps)


def display(msg, info_frame_, cam, recv_fps):
    proc_frame = np.frombuffer(msg[0], dtype=info_frame_.dtype)
    proc_frame.shape = info_frame_.shape
    proc_frame = np.flip(proc_frame, axis=0)
    cam.send(proc_frame)
    proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Webcam', proc_frame)
    cv2.waitKey(1)
    # recv_fps()


def pull(info_frame_, base_fps_):
    cam = pyvirtualcam.Camera(width=info_frame_.shape[1], height=info_frame_.shape[0], fps=base_fps_, delay=0)

    context1 = zmq.Context()
    socket1 = context1.socket(zmq.PULL)
    socket1.bind('tcp://*:5556')

    loop = ioloop.IOLoop.instance()
    stream = zmqstream.ZMQStream(socket1, loop)

    recv_fps = FPS('Recv:')

    stream.on_recv(lambda msg: display(msg, info_frame_, cam, recv_fps))
    loop.start()


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def calc_origin(hmap):
    origin = ndimage.measurements.center_of_mass(hmap)
    origin_inv = origin[::-1]
    return origin_inv[0] / 32, 1 - (origin_inv[1] / 32)


indexes_y = np.arange(5, 21).reshape((4, 4))
# indexes_y = np.arange(1, 21).reshape((5, 4))
indexes_x = indexes_y.T


def calc_hand_data(iv):
    origin = to_vec(calc_origin(iv.hmap[0][..., 9]))
    #     origin = to_vec([iv.uv[9][1] / 32, 1 - (iv.uv[9][0]) / 32])

    #     UV instead of XY
    #     joints = list([to_vec([*(uv[::-1]/32), xyz[2]]) for xyz, uv in zip(iv.xyz, iv.uv)])
    #     joints = list([to_vec([*(uv[::-1]/32), xyz[2]]) for xyz, uv in zip(iv.xyz, xy)])

    joints = list([to_vec(xyz) for xyz in iv.xyz.tolist()])  # tolist, because 0mq can't work with np.float

    def calc_dists(indexes):
        return np.array([
            np.linalg.norm(iv.xyz[i[1:], :2] - iv.xyz[i[:-1], :2]) * 32 /
            np.linalg.norm(iv.uv[i[1:]] - iv.uv[i[:-1]])
            for i in indexes
        ])

    dists_x = filter_nan_inf(calc_dists(indexes_x))
    dists_y = filter_nan_inf(calc_dists(indexes_y))

    if dists_x.size == 0 or dists_y.size == 0:
        return None

    dist_x = np.mean(dists_x)
    dist_y = np.mean(dists_y)

    # palm_size = sum(np.linalg.norm(iv.xyz[0] - iv.xyz[i]) for i in [1, 5, 9, 13, 17])

    u09 = unit_vector(iv.xyz[9, :2] - iv.xyz[0, :2])
    #     horiz = abs(2 * np.arcsin(u09[0]) / np.pi)
    vert = abs(2 * np.arcsin(u09[1]) / np.pi)

    return origin, joints, dist_x, dist_y, vert


def crop(frame):
    height, width, _ = frame.shape
    size = min(int(width / 2), height)
    margin_h = int((width - 2 * size) / 2)
    margin_v = int((height - size) / 2)
    return frame[margin_v:size + margin_v, margin_h:width - margin_h]


def main():
    model = ModelPipeline()

    webcam = WebcamCapture()

    base_fps = webcam.get(cv2.CAP_PROP_FPS)
    print('Base FPS:', base_fps)

    info_frame = crop(webcam.read())

    mp.Process(target=pull, args=(info_frame, base_fps)).start()

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind('tcp://*:5555')

    height, width, _ = info_frame.shape

    hand_data_keys = ['origin', 'joints', 'distX', 'distY', 'vert']

    fps_send = FPS('Send:')
    while True:
        frame_large = webcam.read_rgb()
        frame_large = crop(frame_large)

        frame_large_l = frame_large[:, :width // 2]
        frame_large_r = frame_large[:, width // 2:]

        frame_l = imresize(frame_large_l, (128, 128))
        frame_r = imresize(frame_large_r, (128, 128))

        # iv - intermediate values
        ivl, _ = model.process(np.flip(frame_l, axis=1))
        ivr, _ = model.process(frame_r)

        hand_data_l = calc_hand_data(ivl)
        hand_data_r = calc_hand_data(ivr)

        if hand_data_l is not None and hand_data_r is not None:
            socket.send_json({
                'dataL': dict(zip(hand_data_keys, hand_data_l)),
                'dataR': dict(zip(hand_data_keys, hand_data_r)),
                'frameWidth': frame_large.shape[1],
                'frameHeight': frame_large.shape[0],
            }, zmq.SNDMORE)
            socket.send(np.flip(frame_large, axis=0).tobytes())

            fps_send()


if __name__ == '__main__':
    main()
