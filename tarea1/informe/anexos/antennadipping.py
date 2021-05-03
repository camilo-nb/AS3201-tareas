import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("../antdip_AE2021A.xlsx", engine='openpyxl')

_secz = df.iloc[:-1, 3]
lndP = df.iloc[:-1, 7]

p = np.poly1d(np.polyfit(_secz, lndP, 1)) 
print("tau_w = {}".format(p.coefficients[0]))
print("ln(deltaW) = -sec(z) * {} + {}". format(*p.coefficients))

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))

ax.plot(_secz, p(_secz), label="y = {1:.3f} + x * {0:.3f}".format(*p.c), color="tab:blue")
ax.plot(_secz, lndP, label="Datos", marker='x', fillstyle="none", linestyle="", color="tab:red")
ax.legend()

ax.set_xlabel(r"$-\sec(\varphi)$")
ax.set_ylabel(r"$\ln(\Delta W)$")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)

fig.savefig("../informe/taufit.pdf")
plt.show()
