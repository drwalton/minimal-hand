using System;
using UnityEditor.Animations;
using UnityEngine;
using UnityEngine.UI;

[Serializable]
public class SliderController
{
    public Slider slider;
    public AnimatorController controller;

    public void Deconstruct(out Slider slider, out AnimatorController controller)
    {
        slider = this.slider;
        controller = this.controller;
    }
}

public class SetControllerSlider : MonoBehaviour
{
    public float sliderSpeed = 0.0043f;
    public Client client;
    public SliderController[] sliderController;
    public Animator animator;

    private void Start()
    {
        sliderController[0].slider.value = 1f;
        animator.runtimeAnimatorController = sliderController[0].controller;
    }

    private void Update()
    {
        if (client.rightHand is null) return;

        foreach (var (slider, controller) in sliderController)
        {
            if (slider.value.Equals(1f)) continue;

            if (client.rightHand.show.Equals(1f) &&
                client.rightHand.indexDist > client.handParams.indexDistMin &&
                RectTransformUtility.RectangleContainsScreenPoint(slider.targetGraphic.rectTransform,
                    Camera.main.WorldToScreenPoint(client.rightHand.spheres[8].transform.position), Camera.main))
            {
                slider.value = Mathf.Min(1f, slider.value + sliderSpeed);

                if (slider.value.Equals(1f))
                {
                    animator.runtimeAnimatorController = controller;

                    foreach (var (slider1, _) in sliderController)
                    {
                        if (slider1 != slider)
                        {
                            slider1.value = 0f;
                        }
                    }
                }
            }
            else
            {
                slider.value = Mathf.Max(0f, slider.value - sliderSpeed);
            }
        }
    }
}