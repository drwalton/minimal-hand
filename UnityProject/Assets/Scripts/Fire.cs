using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fire
{
    public readonly Hand hand;
    public readonly GameObject fire;

    public Fire(Hand hand, GameObject fire)
    {
        this.hand = hand;
        this.fire = fire;
    }

    public void Draw()
    {
        if (hand.show.Equals(1f) && hand.indexDist > hand.hparams.indexDistMin)
        {
            fire.SetActive(true);
            fire.transform.position = hand.spheres[8].transform.position;
        }
        else
        {
            fire.SetActive(false);
        }
    }
}
