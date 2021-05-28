import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from vtan import v0, R0, R, vtan

w0 = v0/R0.to(u.km)

w = w0+vtan/R.to(u.km)

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, vtan, c='r', lw=1)
ax.set_xlabel(r"$R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\tan$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/vtan.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, w, c='r', lw=1)
ax.set_xlabel(r"$R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$\omega$ [rad/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/w.pdf")
plt.show()
