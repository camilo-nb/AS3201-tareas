#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

from curvaderotacion import R0, R, l, b_v_maximorum

def Z(l: float, b: float) -> float:
    return R0*np.cos(l*np.pi/180.)*np.tan(b*np.pi/180.)

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, Z(l, b_v_maximorum), c='k', lw=0.25, marker='s', markersize=1, mfc="none", markeredgewidth=0.25)
ax.set_xlabel(r"$R$ [kpc]")
ax.set_ylabel(r"$Z$ [kpc]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=False, right=True)
axl = ax.secondary_xaxis("top", functions=(lambda R: (180./np.pi*np.arcsin(R/R0.value)-360)%360, lambda l: R0.value*np.sin(l*np.pi/180.)))
axl.set_xlabel(r"$l$ [°]")
axl.tick_params(direction="in", top=True)
fig.savefig("../informe/rsc/Z.pdf")
plt.show()
