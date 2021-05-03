import glob
import itertools
import numpy as np
import scipy.optimize
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
ax.scatter(liirad, biirad, marker='.', color="tab:red")
fig.savefig("../informe/lb.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.scatter(lii, bii, color="tab:red")
ax.set_xlabel(r"lii")
ax.set_ylabel(r"bii")
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
        axs[n-1].plot(v[5*j+i], T[5*j+i], color="green", drawstyle="steps-mid", linestyle="-", linewidth=0.5)
        axs[n-1].plot(v[5*j+i], gaussian(v[5*j+i], *popt), color="red", drawstyle="default", linestyle="-", linewidth=0.5)
        Tmax[i, j] = T[5*j+i].max()
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
for x in lii[0:5][[1, 2, 3]]:
    axlii.axvline(x, color="darkred", linestyle="--", linewidth=0.5)
axlii.set_xlabel(r"$l^{II} [\degree]$", color="red")
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
for x in bii[0:5][[0, 2, 4]]:
    axbii.axvline(x, color="darkgreen", linestyle=":", linewidth=0.5)
axbii.xaxis.tick_top()
axbii.yaxis.tick_right()
axbii.set_xlabel(r"$b^{II} [\degree]$", color="green")
ylabelaxbii = axbii.set_ylabel(r"$T_{max}$ [K]", color="green")
ylabelaxbii.set_rotation(-90)
axbii.xaxis.set_label_position('top') 
axbii.yaxis.set_label_position('right') 
axbii.tick_params(axis='x', colors="green")
axbii.tick_params(axis='y', colors="green")
axbii.yaxis.set_tick_params(rotation=-90)
axbii.yaxis.set_label_coords(1.11, 0.5)
axbii.tick_params(direction="in")

fig.savefig("../informe/tmax.pdf")
plt.show()