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
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("../informe/specfit{}.pdf".format(j+1))
    plt.show()