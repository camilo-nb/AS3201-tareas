#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.optimize import curve_fit

from curvaderotacion import R, vrot

G = 4.302e-6*u.kpc*u.km**2/u.s**2/u.Msun

vrot = -vrot.value
R = -R.value

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

popt_point, pcov_point = curve_fit(lambda R, M0: vel(R, pointlike_mass_model(M0)), R, vrot)
perr_point = np.sqrt(np.diag(pcov_point))
popt_disk, pcov_disk = curve_fit(lambda R, s: vel(R, uniform_disk_mass_model(R, s)), R, vrot)
perr_disk = np.sqrt(np.diag(pcov_disk))
popt_sphere, pcov_sphere = curve_fit(lambda R, rho: vel(R, uniform_sphere_mass_model(R, rho)), R, vrot)
perr_sphere = np.sqrt(np.diag(pcov_sphere))
popt_disk_point, pcov_disk_point = curve_fit(lambda R, s, M0: vel(R, uniform_disk_and_pointlike_mass_model(R, s, M0)), R, vrot)
perr_disk_point = np.sqrt(np.diag(pcov_disk_point))
popt_sphere_point, pcov_sphere_points = curve_fit(lambda R, rho, M0: vel(R, uniform_sphere_and_pointlike_mass_model(R, rho, M0)), R, vrot)
perr_sphere_point = np.sqrt(np.diag(pcov_sphere_points))

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, pointlike_mass_model(*popt_point)), c='g', lw=1, label="Point")
ax.fill_between(R, -vel(R, pointlike_mass_model(*(popt_point+perr_point))), -vel(R, pointlike_mass_model(*(popt_point-perr_point))), facecolor='g', alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="lower right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/point.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, uniform_disk_mass_model(R, *popt_disk)), c='g', lw=1, label="Uniform Disk")
ax.fill_between(R, -vel(R, uniform_disk_mass_model(R, *(popt_disk+perr_disk))), -vel(R, uniform_disk_mass_model(R, *(popt_disk-perr_disk))), facecolor='g', alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="lower right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/disk.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, uniform_sphere_mass_model(R, *popt_sphere)), c='g', lw=1, label="Uniform Sphere")
ax.fill_between(R, -vel(R, uniform_sphere_mass_model(R, *(popt_sphere+perr_sphere))), -vel(R, uniform_sphere_mass_model(R, *(popt_sphere-perr_sphere))), facecolor='g', alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="lower right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/sphere.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, uniform_disk_and_pointlike_mass_model(R, *popt_disk_point)), c='g', lw=1, label="Uniform Disk + Point")
ax.fill_between(R, -vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point+perr_disk_point))), -vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point-perr_disk_point))), facecolor='g', alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="lower right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/diskpoint.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, uniform_sphere_and_pointlike_mass_model(R, *popt_sphere_point)), c='g', lw=1, label="Uniform Sphere + Point")
ax.fill_between(R, -vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point+perr_sphere_point))), -vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point-perr_sphere_point))), facecolor='g', alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="lower right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/spherepoint.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.plot(R, -vel(R, uniform_sphere_and_pointlike_mass_model(R, *popt_sphere_point)), c="tab:orange", lw=1, label="Uniform Sphere + Point")
ax.fill_between(R, -vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point+perr_sphere_point))), -vel(R, uniform_sphere_and_pointlike_mass_model(R, *(popt_sphere_point-perr_sphere_point))), facecolor="tab:orange", alpha=0.25)
ax.plot(R, -vel(R, uniform_sphere_mass_model(R, *popt_sphere)), c="tab:brown", lw=1, label="Uniform Sphere")
ax.fill_between(R, -vel(R, uniform_sphere_mass_model(R, *(popt_sphere+perr_sphere))), -vel(R, uniform_sphere_mass_model(R, *(popt_sphere-perr_sphere))), facecolor="tab:brown", alpha=0.25)
ax.plot(R, -vel(R, uniform_disk_and_pointlike_mass_model(R, *popt_disk_point)), c="tab:green", lw=1, label="Uniform Disk + Point")
ax.fill_between(R, -vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point+perr_disk_point))), -vel(R, uniform_disk_and_pointlike_mass_model(R, *(popt_disk_point-perr_disk_point))), facecolor="tab:green", alpha=0.25)
ax.plot(R, -vel(R, uniform_disk_mass_model(R, *popt_disk)), c="tab:purple", lw=1, label="Uniform Disk")
ax.fill_between(R, -vel(R, uniform_disk_mass_model(R, *(popt_disk+perr_disk))), -vel(R, uniform_disk_mass_model(R, *(popt_disk-perr_disk))), facecolor="tab:purple", alpha=0.25)
ax.plot(R, -vel(R, pointlike_mass_model(*popt_point)), c="tab:blue", lw=1, label="Point")
ax.fill_between(R, -vel(R, pointlike_mass_model(*(popt_point+perr_point))), -vel(R, pointlike_mass_model(*(popt_point-perr_point))), facecolor="tab:blue", alpha=0.25)
ax.plot(R, -vrot, c='r', ls="", marker=".", markersize=1, label="Data")
ax.legend(loc="upper right")
ax.set_xlabel(r"$-R_{\odot}\sin\,l$ [kpc]")
ax.set_ylabel(r"$v_\mathrm{rot}$ [km/s]")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/rsc/massmodels.pdf")
plt.show()
