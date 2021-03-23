using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Hand
{
    public readonly HandParams hparams;
    public readonly float xMult;

    public readonly GameObject spheresParent;
    public readonly GameObject[] spheres;
    public readonly LineRenderer[] lineRenderers;
    public readonly Queue<float> waySumQueue;
    public float show = 0f;
    public float indexDist;

    public Hand(HandParams hparams, float xMult)
    {
        this.hparams = hparams;
        this.xMult = xMult;

        spheresParent = new GameObject("Hand");
        spheres = new GameObject[HandParams.JointsNum];
        lineRenderers = new LineRenderer[HandParams.JointsNum];
        foreach (var i in Enumerable.Range(0, HandParams.JointsNum))
        {
            spheres[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            spheres[i].transform.parent = spheresParent.transform;
            spheres[i].transform.localScale =
                new Vector3(hparams.sphereScale, hparams.sphereScale, hparams.sphereScale);
            lineRenderers[i] = spheres[i].AddComponent<LineRenderer>();
            lineRenderers[i].widthMultiplier = 0.1f;
        }

        waySumQueue = new Queue<float>();
    }

    public void Process(Data.HandData data)
    {
        float scaleX = data.vert * data.distX + (1 - data.vert) * data.distY;
        float scaleY = data.vert * data.distY + (1 - data.vert) * data.distX;
        float dist = (scaleX + scaleY) / 2;

        // Calc total way and move the spheres
        var waySum = 0f;
        foreach (var i in Enumerable.Range(0, HandParams.JointsNum))
        {
            var target = new Vector3(
                (+data.joints[i].x / scaleX + data.origin.x) * (Client.frustumWidth / 2) * xMult,
                (-data.joints[i].y / scaleY + (data.origin.y - 0.5f)) * Client.frustumHeight,
                hparams.zAlpha * data.joints[i].z + hparams.zBeta * dist
            );

            var actualTarget = Vector3.Lerp(spheres[i].transform.position, target,
                Vector3.Distance(spheres[i].transform.position, target) * hparams.speed);

            waySum += Vector3.Distance(spheres[i].transform.position, actualTarget);

            spheres[i].transform.position = actualTarget;
        }
        waySumQueue.Enqueue(waySum);
        //

        // Calc show
        var avgFramesToHide = Mathf.Max(1f, hparams.avgSecondsToHide / Time.deltaTime);
        while (waySumQueue.Count > avgFramesToHide) waySumQueue.Dequeue();
        var avgWaySum = waySumQueue.Average();
        if (avgWaySum > hparams.maxAvgWaySum || dist < hparams.minDist || dist > hparams.maxDist)
        {
            show = 0f;
        }
        else
        {
            show = Mathf.Min(show + hparams.showSpeed / avgWaySum, 1f);
        }
        spheresParent.SetActive(show.Equals(1f));
        //

        // Move lines
        foreach (
            var i in Enumerable.Empty<int>()
                .Concat(Enumerable.Range(0, 4))
                .Concat(Enumerable.Range(5, 3))
                .Concat(Enumerable.Range(9, 3))
                .Concat(Enumerable.Range(13, 3))
                .Concat(Enumerable.Range(17, 3))
        )
        {
            lineRenderers[i].SetPosition(0, spheres[i].transform.position);
            lineRenderers[i].SetPosition(1, spheres[i + 1].transform.position);
        }
        //

        
        indexDist =
            Vector3.Distance(spheres[8].transform.position, spheres[12].transform.position) +
            Vector3.Distance(spheres[8].transform.position, spheres[16].transform.position) +
            Vector3.Distance(spheres[8].transform.position, spheres[20].transform.position);
    }
}