from pyprocessmacro import Process
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./TestCases/Data_Model12.csv")
varlist = open("./Tests/Data/Varlist_Model12.txt").read().split(",")
kwargs = {i: i for i in varlist if "m" not in i}
df["ZOO"] = df["z"]
df["KA"] = df["w"]
kwargs["m"] = ["m1", "m2"]
kwargs["z"] = "ZOO"
kwargs["w"] = "KA"
p = Process(data=df, model=12, **kwargs, seed=14568743, hc3=True)
jn_regions = p.compute_jn_region("direct", "KA")
print(p.spotlight_direct_effect({"KA": [*jn_regions[0], *jn_regions[1]],
                                            "ZOO":[0]}))
jn_regions = p.compute_jn_region("indirect", "KA", med_name="m1")
print(p.spotlight_indirect_effect("m1", {"KA": [*jn_regions[0], *jn_regions[1]],
                                            "ZOO":[0]}))

g = p.plot_conditional_direct_effects(x="KA", modval={"KA":np.linspace(-2, 2)})

for r in jn_regions:
    for v in r:
        g.ax.axvline(v)
g.ax.set_xlim(-1, 1)
plt.show()
