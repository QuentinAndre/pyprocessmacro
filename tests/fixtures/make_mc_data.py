"""
Make tests/Data/Data_MC.csv: synthetic data with a four-level categorical X and a three-level categorical W
for the multicategorical comparison files (#17). Deterministic; rerunning rewrites the same file.

Columns: xcat (1 to 4, unequal groups), x (continuous), w (continuous), wcat (1 to 3), m and m2 (mediators),
y (continuous), y2 (binary). Group effects, interactions with the moderators and a direct effect are built in so
that every relative effect PROCESS reports is far from zero.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "Data_MC.csv")


def make(n=1000, seed=20260926):
    rng = np.random.default_rng(seed)
    xcat = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.25, 0.25, 0.15])
    wcat = rng.choice([1, 2, 3], size=n, p=[0.4, 0.35, 0.25])
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    xg = np.array([0.0, 0.5, 1.0, 0.3])[xcat - 1]      # group effect of X on M
    xgw = np.array([0.0, 0.3, -0.2, 0.1])[xcat - 1]    # X-group by W interaction on M
    wg = np.array([0.0, 0.4, -0.3])[wcat - 1]          # W-group main effect
    wgx = np.array([0.0, 0.35, -0.25])[wcat - 1]       # X by W-group interaction
    m = xg + 0.3 * w + xgw * w + 0.4 * x + 0.2 * x * w + wg + wgx * x + rng.normal(size=n)
    m2 = 0.6 * xg + 0.3 * x + 0.2 * w + rng.normal(size=n)
    xgd = np.array([0.0, 0.2, 0.4, -0.1])[xcat - 1]    # direct effect of the X groups on Y
    y = (0.5 * m + 0.3 * m2 + xgd + 0.3 * w + 0.2 * m * w + 0.15 * xgd * w + 0.3 * x + 0.2 * x * w
         + wg + wgx * x + 0.15 * m * wg + rng.normal(size=n))
    logit = -0.2 + 0.6 * m + 0.4 * xgd + 0.3 * x + 0.2 * w
    y2 = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(dict(xcat=xcat, x=x, w=w, wcat=wcat, m=m, m2=m2, y=y, y2=y2))


if __name__ == "__main__":
    frame = make()
    frame.to_csv(OUT, index=False, float_format="%.10g")
    print("wrote", OUT, frame.shape)
