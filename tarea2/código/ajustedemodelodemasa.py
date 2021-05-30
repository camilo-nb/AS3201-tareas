#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit

from curvaderotacion import R, vrot

G = 4.302e-6*u.kpc*u.km**2/u.s**2/u.Msun

vrot = vrot.value
R = R.value

def vel(R: float, M: float) -> float:
    return np.sqrt(G.value*M/R)

def pointlike_mass_model(M0: float) -> float:
    """`M0` point mass in the center of the Galaxy."""
    return M0

def uniform_disk_mass_model(R: float, s: float) -> float:
    """`s` uniform superficial mass density."""
    return np.pi*R**2*s

def uniform_sphere_mass_model(R: float, rho: float) -> float:
    """`rho` uniform volumetric mass density."""
    return 4/3*np.pi*R**3*rho

def uniform_disk_and_pointlike_mass_model(R: float, s: float, M0: float) -> float:
    return uniform_disk_mass_model(R, s) + pointlike_mass_model(M0)

def uniform_sphere_and_pointlike_mass_model(R: float, rho: float, M0: float) -> float:
    return uniform_sphere_mass_model(R, rho) + pointlike_mass_model(M0)

popt_point, pcov_point = curve_fit(lambda R, M0: vel(R, pointlike_mass_model(M0)), R, vrot, check_finite=False)
perr_point = np.sqrt(np.diag(pcov_point))
popt_disk, pcov_disk = curve_fit(lambda R, s: vel(R, uniform_disk_mass_model(R, s)), R, vrot, check_finite=False)
perr_disk = np.sqrt(np.diag(pcov_disk))
popt_sphere, pcov_sphere = curve_fit(lambda R, rho: vel(R, uniform_sphere_mass_model(R, rho)), R, vrot, check_finite=False)
perr_sphere = np.sqrt(np.diag(pcov_sphere))
popt_disk_point, pcov_disk_point = curve_fit(lambda R, s, M0: vel(R, uniform_disk_and_pointlike_mass_model(R, s, M0)), R, vrot, check_finite=False)
perr_disk_point = np.sqrt(np.diag(pcov_disk_point))
popt_sphere_point, pcov_sphere_points = curve_fit(lambda R, rho, M0: vel(R, uniform_sphere_and_pointlike_mass_model(R, rho, M0)), R, vrot, check_finite=False)
perr_sphere_point = np.sqrt(np.diag(pcov_sphere_points))

print("M0 =", popt_point, "+-", perr_point, "RMS =", np.sqrt(np.mean((vrot-vel(R, pointlike_mass_model(*popt_point)))**2)))
print("s0 =", popt_disk, "+-", perr_disk, "RMS =", np.sqrt(np.mean((vrot-vel(R, uniform_disk_mass_model(R, *(popt_disk+perr_disk))))**2)))
print("rho0 =", popt_sphere, "+-", perr_sphere, "RMS =", np.sqrt(np.mean((vrot-vel(R, uniform_sphere_mass_model(R, *popt_sphere)))**2)))
print("s0 M0 =", popt_disk_point, "+-", perr_disk_point, "RMS =", np.sqrt(np.mean((vrot-vel(R, uniform_disk_and_pointlike_mass_model(R, *popt_disk_point)))**2)))
print("rho0 M0 =", popt_sphere_point, "+-", perr_sphere_point, "RMS =", np.sqrt(np.mean((vrot-vel(R, uniform_sphere_and_pointlike_mass_model(R, *popt_sphere_point)))**2)))

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, vrot, c='k', lw=0.25, marker='s', markersize=1, mfc="none", markeredgewidth=0.25, label="Datos")
ax.plot(R, vel(R, pointlike_mass_model(*popt_point)), c="tab:red", lw=0.5, label="Punto")
ax.fill_between(R, vel(R, pointlike_mass_model(*(popt_point+perr_point))), vel(R, pointlike_mass_model(*(popt_point-perr_point))), facecolor="tab:red", alpha=0.25)
ax.plot(R, vel(R, uniform_sphere_and_pointlike_mass_model(R, *popt_sphere_point)), c="tab:purple", lw=0.5, label="Esfera + Punto")
ax.fill_between(R, vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point+perr_sphere_point))), vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point-perr_sphere_point))), facecolor="tab:purple", alpha=0.25)
ax.plot(R, vel(R, uniform_sphere_mass_model(R, *popt_sphere)), c="tab:orange", lw=0.5, label="Esfera")
ax.fill_between(R, vel(R, uniform_sphere_mass_model(R, *(popt_sphere+perr_sphere))), vel(R, uniform_sphere_mass_model(R, *(popt_sphere-perr_sphere))), facecolor="tab:orange", alpha=0.25)
ax.plot(R, vel(R, uniform_disk_and_pointlike_mass_model(R, *popt_disk_point)), c="tab:blue", lw=0.5, label="Disco + Punto")
ax.fill_between(R, vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point+perr_disk_point))), vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point-perr_disk_point))), facecolor="tab:blue", alpha=0.25)
ax.plot(R, vel(R, uniform_disk_mass_model(R, *popt_disk)), c="tab:green", lw=0.5, label="Disco")
ax.fill_between(R, vel(R, uniform_disk_mass_model(R, *(popt_disk+perr_disk))), vel(R, uniform_disk_mass_model(R, *(popt_disk-perr_disk))), facecolor="tab:green", alpha=0.25)
ax.legend(loc="lower right")
ax.set_xlabel(r"$R$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/massmodels.pdf")
plt.show()
