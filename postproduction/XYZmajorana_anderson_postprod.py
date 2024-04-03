import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import h5py
from XYZmajorana_class import QuantileCalc, ErrorCalc, spectrum



# %% Reading system size 10
with open('params_XYZ_L10.txt', 'r') as f:
    Nsamples10 = len(f.readlines())

for i, file in enumerate(os.listdir('XYZmaj_L10_anderson')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L10_anderson', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            marker = datanode[()]
            if i == 0:
                marker10 = np.zeros((marker.shape[0], Nsamples10))
                gamma10 = datanode.attrs['gamma']
                X10 = datanode.attrs['X']
                Y10 = datanode.attrs['Y']
                Z10 = datanode.attrs['Z']
                L10 = datanode.attrs['L']
            marker10[:, i] = marker

avg_marker10 = np.mean(marker10, axis=1)
std_marker10 = np.std(marker10, axis=1)
median_marker10 = np.median(marker10, axis=1)
q25_marker10, q75_marker10 = QuantileCalc(marker10, 0.75, 0.01)
# error_down10, error_up10 = ErrorCalc(marker10, 0.95, 1000, 0.01)
# error10 = np.array([np.abs(error_down10 - median_marker10), np.abs(error_up10-median_marker10)])
print('L10 done')



# Inset Probability distribution
nbins = 10
nu = marker10
Pnu10, bins = np.histogram(nu, range=(-0.03, 0.03))
nu10 = 0.5 * (bins[1:] + bins[:-1])
    


# %% Figures
# plt.style.use('./stylesheets/prb.mplstyle')
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
yellow = (1, 0.75, 0)
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C', '#00BFFF']
shadeblue = '#00BFFF'
inset_color =['#DC143C', '#00BFFF','#9A32CD','#EE3A8C']

fig1, ax = plt.subplots(figsize=(6, 5))
ax.plot(nu10, Pnu10, marker='.', color=axcolour[1])
ax.set_xlabel("$\\nu$", fontsize=16)
ax.set_ylabel("$P(\\nu)$", fontsize=16)




# plt.tight_layout()
plt.savefig("XYZmaj_anderson.pdf", bbox_inches="tight")
plt.show()















