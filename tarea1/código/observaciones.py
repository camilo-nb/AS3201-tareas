#!/usr/bin/python3

import glob
import itertools
import numpy as np
import scipy.optimize
import scipy.integrate
import astropy.stats
import matplotlib.pyplot as plt

paths = sorted(glob.glob("../sec_mierc_sem1_2021/sdf*"))

lii = np.zeros(len(paths))
bii = np.zeros(len(paths))
v = np.zeros((len(paths), 364-109+1))
T = np.zeros((len(paths), 364-109+1))
for i, path in enumerate(paths):
    with open(path) as f:
        [_, lii[i]], [_, bii[i]] = np.genfromtxt(itertools.islice(f, 22, 24, None))
        v[i], T[i] = np.genfromtxt(path, unpack=True, skip_header=108)

liirad = np.radians((lii - 180) % 360 - (360 - 180))
biirad = np.radians((bii - 180) % 360 - (360 - 180))

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(3.25, 3.25/2))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(True)
ax.scatter(liirad, biirad, marker='.', color="red")
fig.savefig("../informe/lb.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.scatter(lii, bii, color="red")
ax.set_xlabel(r"$l^{II} [\degree]$")
ax.set_ylabel(r"$b^{II} [\degree]$")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
fig.savefig("../informe/cruz.pdf")
plt.show()

def gaussian(x, height, center, width):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)

Tmax = np.zeros((5, 3))
for j in range(3):
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(3.25, 3.25))
    axs = axs.flatten()
    for i in range(9):
        axs[i].set_axis_off()
    for i, n in enumerate([2, 4, 5, 6, 8]):
        axs[n-1].set_axis_on()
        axs[n-1].set_ylim((-1,32))
        axs[n-1].yaxis.set_tick_params(rotation=90)
        axs[n-1].tick_params(direction="in", top=True, right=True)
        popt, pcov = scipy.optimize.curve_fit(gaussian, v[5*j+i], T[5*j+i], p0=[20, 10, 1])
        axs[n-1].plot(v[5*j+i], T[5*j+i], c="green", drawstyle="steps-mid", ls="-", lw=0.5)
        axs[n-1].plot(v[5*j+i], gaussian(v[5*j+i], *popt), c="red", ls="-", lw=0.5)
        Tmax[i, j] = T[5*j+i].max()
        plt.text(0.3, 0.85, r"$l^{II}=$"+f"{lii[5*j+i]:.2f}", fontsize=5,
                 ha='center', va='center', transform=axs[n-1].transAxes)
        plt.text(0.3, 0.75, r"$b^{II}=$"+f"{bii[5*j+i]:.2f}", fontsize=5,
                 ha='center', va='center', transform=axs[n-1].transAxes)
    axs[8-1].set_xlabel(r"Velocidad [km/s]")
    axs[4-1].set_ylabel(r"Temperatura [K]")
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("../informe/specfit{}.pdf".format(j+1))
    plt.show()

Tmax = Tmax.mean(axis=1)
fig = plt.figure(figsize=(3.25, 3.25))

# https://doi.org/10.1109/MSP.2011.941846
p = np.poly1d(np.polyfit(lii[0:5][[1, 2, 3]], np.log(Tmax[[1, 2, 3]]), 2))
heightlii = np.exp(p.c[2]-p.c[1]*p.c[1]/(4*p.c[0]))
centerlii = -p.c[1]/(2*p.c[0])
widthlii = np.sqrt(-1/(2*p.c[0]))

xlii = np.linspace(lii.min()-0.075, lii.max()+0.075)
ylii = gaussian(xlii, heightlii, centerlii, widthlii)

axlii = fig.add_subplot(111, label=r"$l^{II}$")
axlii.plot(xlii, ylii, color="red")
axlii.plot(lii[0:5][[1, 2, 3]], Tmax[[1, 2, 3]], marker='x', color="darkred", linestyle="")
axlii.set_xlabel(r"$l^{II} [\degree], b^{II}=$"+f"{bii[2]:.2f}"+r"$\degree$", c="red")
axlii.set_ylabel(r"$T_{max} [K]$", color="red")
axlii.tick_params(axis='x', colors="red")
axlii.tick_params(axis='y', colors="red")
axlii.yaxis.set_tick_params(rotation=90)
axlii.tick_params(direction="in")

p = np.poly1d(np.polyfit(bii[0:5][[0, 2, 4]], np.log(Tmax[[0, 2, 4]]), 2))
heightbii = np.exp(p.c[2]-p.c[1]*p.c[1]/(4*p.c[0]))
centerbii = -p.c[1]/(2*p.c[0])
widthbii = np.sqrt(-1/(2*p.c[0]))
xbii = np.linspace(bii.min()-0.075, bii.max()+0.075)
ybii = gaussian(xbii, heightbii, centerbii, widthbii)
axbii = fig.add_subplot(111, label=r"$b^{II}$", frame_on=False)
axbii.plot(xbii, ybii, color="green")
axbii.plot(bii[0:5][[0, 2, 4]], Tmax[[0, 2, 4]], marker='x', color="darkgreen", linestyle="")
axbii.xaxis.tick_top()
axbii.yaxis.tick_right()
axbii.set_xlabel(r"$b^{II} [\degree], l^{II}=$"+f"{lii[2]:.2f}"+r"$\degree$", c="green")
ylabelaxbii = axbii.set_ylabel(r"$T_{max}$ [K]", color="green")
ylabelaxbii.set_rotation(-90)
axbii.xaxis.set_label_position('top') 
axbii.yaxis.set_label_position('right') 
axbii.tick_params(axis='x', colors="green")
axbii.tick_params(axis='y', colors="green")
axbii.yaxis.set_tick_params(rotation=-90)
axbii.yaxis.set_label_coords(1.1175, 0.5)
axbii.tick_params(direction="in")

fig.savefig("../informe/tmax.pdf")
plt.show()

print("Data: Tmax = {}, lii = {}". format(Tmax[2], lii[2]))
print("Model: Tmax = {}, lii = {}". format(ylii.max(), xlii[np.argmax(ylii)]))
print("Data: Tmax = {}, bii = {}". format(Tmax[2], bii[2]))
print("Model: Tmax = {}, bii = {}". format(ybii.max(), xbii[np.argmax(ybii)]))

Tint = -scipy.integrate.simpson(T, v, axis=1).reshape((3, 5)).T.mean(axis=1)
fig = plt.figure(figsize=(3.25, 3.25))
p = np.poly1d(np.polyfit(lii[0:5][[1, 2, 3]], np.log(Tint[[1, 2, 3]]), 2))
heightlii = np.exp(p.c[2]-p.c[1]*p.c[1]/(4*p.c[0]))
centerlii = -p.c[1]/(2*p.c[0])
widthlii = np.sqrt(-1/(2*p.c[0]))
xlii = np.linspace(lii.min()-0.075, lii.max()+0.075)
ylii = gaussian(xlii, heightlii, centerlii, widthlii)
axlii = fig.add_subplot(111, label=r"$l^{II}$")
axlii.plot(xlii, ylii, color="red")
axlii.plot(lii[0:5][[1, 2, 3]], Tint[[1, 2, 3]], marker='x', color="darkred", linestyle="")
axlii.set_xlabel(r"$l^{II} [\degree], b^{II}=$"+f"{bii[2]:.2f}"+r"$\degree$", c="red")
axlii.set_ylabel(r"$T_{int} [K*km/s]$", color="red")
axlii.tick_params(axis='x', colors="red")
axlii.tick_params(axis='y', colors="red")
axlii.yaxis.set_tick_params(rotation=90)
axlii.tick_params(direction="in")
p = np.poly1d(np.polyfit(bii[0:5][[0, 2, 4]], np.log(Tint[[0, 2, 4]]), 2))
heightbii = np.exp(p.c[2]-p.c[1]*p.c[1]/(4*p.c[0]))
centerbii = -p.c[1]/(2*p.c[0])
widthbii = np.sqrt(-1/(2*p.c[0]))
xbii = np.linspace(bii.min()-0.075, bii.max()+0.075)
ybii = gaussian(xbii, heightbii, centerbii, widthbii)
axbii = fig.add_subplot(111, label=r"$b^{II}$", frame_on=False)
axbii.plot(xbii, ybii, color="green")
axbii.plot(bii[0:5][[0, 2, 4]], Tint[[0, 2, 4]], marker='x', color="darkgreen", linestyle="")
axbii.xaxis.tick_top()
axbii.yaxis.tick_right()
axbii.set_xlabel(r"$b^{II} [\degree], l^{II}=$"+f"{lii[2]:.2f}"+r"$\degree$", c="green")
ylabelaxbii = axbii.set_ylabel(r"$T_{int}$ [K*km/s]", color="green")
ylabelaxbii.set_rotation(-90)
axbii.xaxis.set_label_position('top') 
axbii.yaxis.set_label_position('right') 
axbii.tick_params(axis='x', colors="green")
axbii.tick_params(axis='y', colors="green")
axbii.yaxis.set_tick_params(rotation=-90)
axbii.yaxis.set_label_coords(1.1175, 0.5)
axbii.tick_params(direction="in")
fig.savefig("../informe/tint.pdf")
plt.show()

masked = astropy.stats.sigma_clip(T, axis=1, sigma=3, maxiters=None)
rms = np.sqrt((masked**2).mean(axis=1)).reshape((3, 5)).T
meanmasked = astropy.stats.sigma_clip((T[0:5]+T[5:10]+T[10:15])/3, sigma=3, axis=1, maxiters=None)
rmsmean = np.sqrt((meanmasked**2).mean(axis=1))
print("deltaT_A/T_sys =", rmsmean.reshape((5, 1)) / rms)
print("1/sqrt(3) =", 1/np.sqrt(3))
print("sqrt(2/3) =", np.sqrt(2/3))
