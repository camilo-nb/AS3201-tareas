import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

paths = sorted(glob.glob("../sec_mierc_sem1_2021/sdf*"))

liis = np.zeros(len(paths))
biis = np.zeros(len(paths))
for i, path in enumerate(paths):
    with open(path) as f:
        [_, lii], [_, bii] = np.genfromtxt(itertools.islice(f, 22, 24, None))
        liis[i] = lii
        biis[i] = bii

for lii, bii in zip(liis, biis):
    print(lii, bii)

liis = np.radians((liis - 180) % 360 - (360 - 180))
biis = np.radians((biis - 180) % 360 - (360 - 180))

plt.rcParams.update({'font.size': 5})
fig = plt.figure(figsize=(3.25, 3.25/2))
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(True)
ax.scatter(liis, biis, marker='.', color="tab:red")
fig.savefig("../informe/lb.pdf")
plt.show()

plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.25, 3.25))
ax.scatter(liis, biis, color="tab:red")
ax.set_xlabel(r"lii")
ax.set_ylabel(r"bii")
ax.yaxis.set_tick_params(rotation=90)
ax.tick_params(direction="in", top=True, right=True)
plt.show()
