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

for i, file in enumerate(os.listdir('XYZmaj_L10')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L10', file)
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

# %% Reading system size 14
with open('params_XYZ_L14.txt', 'r') as f:
    Nsamples14 = len(f.readlines())

for i, file in enumerate(os.listdir('XYZmaj_L14')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L14', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            marker = datanode[()]
            if i == 0:
                marker14 = np.zeros((marker.shape[0], Nsamples14))
                gamma14 = datanode.attrs['gamma']
                X14 = datanode.attrs['X']
                Y14 = datanode.attrs['Y']
                Z14 = datanode.attrs['Z']
                L14 = datanode.attrs['L']
            marker14[:, i] = marker

avg_marker14 = np.mean(marker14, axis=1)
std_marker14 = np.std(marker14, axis=1)
median_marker14 = np.median(marker14, axis=1)
q25_marker14, q75_marker14 = QuantileCalc(marker14, 0.75, 0.01)
# error_down14, error_up14 = ErrorCalc(marker14, 0.95, 1000, 0.01)
# error14 = np.array([np.abs(error_down14 - median_marker14), np.abs(error_up14-median_marker14)])
print('L14 done')

# %% Reading system size 18
# opdm18 = np.zeros((300, 36, 36))
# nu18 = np.zeros((300,))
# nu18[i] = marker[19]
Nsamples18 = 900
for i, file in enumerate(os.listdir('XYZmaj_L18_1')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L18_1', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            datanode2 = f['opdm_data']
            marker = datanode[()]
            opdm = datanode2[()]
            if i == 0:
                marker18 = np.zeros((marker.shape[0], Nsamples18))
                opdm18 = np.zeros((opdm.shape[0], opdm.shape[1], opdm.shape[2], Nsamples18))
                gamma18 = datanode.attrs['gamma']
                X18 = datanode.attrs['X']
                Y18 = datanode.attrs['Y']
                Z18 = datanode.attrs['Z']
                L18 = datanode.attrs['L']
            marker18[:, i] = marker
            opdm18[:, :, :, i] = opdm
for i, file in enumerate(os.listdir('XYZmaj_L18_2')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L18_2', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            datanode2 = f['opdm_data']
            marker = datanode[()]
            opdm = datanode2[()]
            marker18[:, 300 + i] = marker
            opdm18[:, :, :, 300 + i] = opdm
for i, file in enumerate(os.listdir('XYZmaj_L18_3')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L18_3', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            datanode2 = f['opdm_data']
            marker = datanode[()]
            opdm = datanode2[()]
            marker18[:, 600 + i] = marker
            opdm18[:, :, :, 600 + i] = opdm
                   

avg_marker18 = np.mean(marker18, axis=1)
std_marker18 = np.std(marker18, axis=1)
median_marker18 = np.median(marker18, axis=1)
q25_marker18, q75_marker18 = QuantileCalc(marker18, 0.75, 0.01)
error_down18, error_up18 = ErrorCalc(marker18, 0.95, 1000, 0.01)
error18 = np.array([np.abs(error_down18 - median_marker18), np.abs(error_up18-median_marker18)])
print('L18 done')
# Inset OPDM
# 40, 5
opdm18_1 = opdm18[:, :, :, 5]
spec = np.zeros((4, opdm18_1.shape[1]))
delta = np.zeros((4, ))
list_deltas = [0, 19, 30, 40]
for i, index in enumerate(list_deltas):
    spec[i, :] = spectrum(opdm18_1[index, :, :])[0]
    delta[i] = gamma18[index]


# Inset Probability distribution
list_indices = np.array([26, 22, 18])
list_deltas2 = gamma18[list_indices]
nbins = 10
Pnu18 = np.zeros((nbins, list_indices.shape[0]))
nu18 = np.zeros((nbins, list_indices.shape[0]))
for i, index in enumerate(list_indices):
    nu = marker18[index]
    Pnu18[:, i], bins = np.histogram(nu)
    nu18[:, i] = 0.5 * (bins[1:] + bins[:-1])
    


# %% Figures
# plt.style.use('./stylesheets/prb.mplstyle')
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
yellow = (1, 0.75, 0)
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C', '#00BFFF']
shadeblue = '#00BFFF'
inset_color =['#DC143C', '#00BFFF','#9A32CD','#00C957']

# fig1, ax = plt.subplots(figsize=(6, 5))
# for i in range(list_indices.shape[0]):
#     ax.plot(nu18[:, i], Pnu18[:, i], marker='.', color=axcolour[i], label='{:.2f}'.format(list_deltas2[i]))
# ax.set_xlim(-1, 0)
# ax.set_yscale('log')
# ax.legend(loc='best', frameon=False)
# ax.set_xlabel("$\\nu$", fontsize=16)
# ax.set_ylabel("$P(\\nu)$", fontsize=16)s

# Medians
fig1, ax = plt.subplots(figsize=(6, 5))
# ax.errorbar(gamma10, median_marker10, yerr=error10, color=axcolour[1], barsabove=True)
ax.plot(gamma10, median_marker10, color=axcolour[1], marker='D', markersize=4, label='{}'.format(L10, Nsamples10))
# ax.errorbar(gamma14, median_marker14, yerr=error14, color=axcolour[2], barsabove=True)
ax.plot(gamma14, median_marker14, color=axcolour[2], marker='^', markersize=4, label='{}'.format(L14, Nsamples14))
ax.errorbar(gamma18, median_marker18, yerr=error18, color=axcolour[3], barsabove=True)
ax.plot(gamma18, median_marker18, color=axcolour[3], marker='.', markersize=12, label='{}'.format(L18, Nsamples18))
ax.fill_between(gamma10, q25_marker10, q75_marker10, color=axcolour[1], alpha=0.3)
ax.fill_between(gamma14, q25_marker14, q75_marker14, color=axcolour[2], alpha=0.3)
ax.fill_between(gamma18, q25_marker18, q75_marker18, color=shadeblue, alpha=0.3)
ax.set_xlim([-8, 8])
ax.set_ylim([-1.1, 1.1])
ax.tick_params(which='major', width=0.75, labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='major', length=14,  labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax.tick_params(which='minor', length=7, direction='in', top=True, right=True)
majorsy = [-1, 0, 1]
minorsy = [-0.5, 0.5]
majorsx = [-8, -4, 0, 4, 8]
minorsx = [-6, -2, 2, 6]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
ax.set_ylabel("$\\nu$", fontsize=20)
ax.set_xlabel("$\delta$", fontsize=20)
ax.legend(bbox_to_anchor=(0.4, 0.7), loc='center right', frameon=False, fontsize=16)
ax.text(-2.85, 0.78, '$L$', fontsize=16)
# plt.tight_layout()
# plt.savefig("XYZmaj_marker_median.pdf", bbox_inches="tight")
# plt.show()
# 
# 
left, bottom, width, height = [0.55, 0.11, 0.35, 0.35]
inset_ax1 = ax.inset_axes([left, bottom, width, height])
for i in range(delta.shape[0]):
    inset_ax1.plot(np.arange(opdm18_1.shape[1]), spec[i, :], '.', color=inset_color[i], markersize=3, label='{:.1f}'.format(delta[i]))
inset_ax1.set_xlabel('$\\alpha$')
inset_ax1.set_ylabel('$n_\\alpha$')
inset_ax1.tick_params(which='major', width=0.75, labelsize=10)
inset_ax1.tick_params(which='major', length=7,  labelsize=10)
inset_ax1.tick_params(which='minor', width=0.75)
inset_ax1.tick_params(which='minor', length=3.5)
majorsy = [0, 0.5, 1]
minorsy = [0.25, 0.75]
majorsx = [-5]
minorsx = [-5]
majorsx_str = [""]
inset_ax1.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))
inset_ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
inset_ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
inset_ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
inset_ax1.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
inset_ax1.legend(bbox_to_anchor=(0.5, 0.7), frameon=False, fontsize=10, handletextpad=0.01)
inset_ax1.text(28, 0.7, '$\delta$', fontsize=10)
inset_ax1.text(2, 0.6, '$L=18$', fontsize=10)




# plt.tight_layout()
plt.savefig("XYZmaj_marker_quantiles.pdf", bbox_inches="tight")
plt.show()















