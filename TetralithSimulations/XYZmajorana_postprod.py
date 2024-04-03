import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import h5py




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
q25_marker10 = np.quantile(marker10, 0.10, axis=1)
q75_marker10 = np.quantile(marker10, 0.85, axis=1)




# %% Reading system size 12
with open('params_XYZ_L12.txt', 'r') as f:
    Nsamples12 = len(f.readlines())

for i, file in enumerate(os.listdir('XYZmaj_L12')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L12', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            marker = datanode[()]
            if i == 0:
                marker12 = np.zeros((marker.shape[0], Nsamples12))
                gamma12 = datanode.attrs['gamma']
                X12 = datanode.attrs['X']
                Y12 = datanode.attrs['Y']
                Z12 = datanode.attrs['Z']
                L12 = datanode.attrs['L']
            marker12[:, i] = marker

avg_marker12 = np.mean(marker12, axis=1)
std_marker12 = np.std(marker12, axis=1)
median_marker12 = np.median(marker12, axis=1)
q25_marker12 = np.quantile(marker12, 0.10, axis=1)
q75_marker12 = np.quantile(marker12, 0.85, axis=1)


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
q25_marker14 = np.quantile(marker14, 0.10, axis=1)
q75_marker14 = np.quantile(marker14, 0.85, axis=1)


# %% Reading system size 14
with open('params_XYZ_L16.txt', 'r') as f:
    Nsamples16 = len(f.readlines())

for i, file in enumerate(os.listdir('XYZmaj_L16')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L16', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            marker = datanode[()]
            if i == 0:
                marker16 = np.zeros((marker.shape[0], Nsamples16))
                gamma16 = datanode.attrs['gamma']
                X16 = datanode.attrs['X']
                Y16 = datanode.attrs['Y']
                Z16 = datanode.attrs['Z']
                L16 = datanode.attrs['L']
            marker16[:, i] = marker

avg_marker16 = np.mean(marker16, axis=1)
std_marker16 = np.std(marker16, axis=1)
median_marker16 = np.median(marker16, axis=1)
q25_marker16 = np.quantile(marker16, 0.10, axis=1)
q75_marker16 = np.quantile(marker16, 0.85, axis=1)
#

# # %% Reading system size 18
with open('params_XYZ_L18.txt', 'r') as f:
    #Nsamples18 = len(f.readlines())
    Nsamples18 = 84
for i, file in enumerate(os.listdir('XYZmaj_L18')):
    if file.endswith('h5'):
        file_path = os.path.join('XYZmaj_L18', file)
        with h5py.File(file_path, 'r') as f:
            datanode = f['marker_data']
            marker = datanode[()]
            if i == 0:
                marker18 = np.zeros((marker.shape[0], Nsamples18))
                gamma18 = datanode.attrs['gamma']
                X18 = datanode.attrs['X']
                Y18 = datanode.attrs['Y']
                Z18 = datanode.attrs['Z']
                L18 = datanode.attrs['L']
            marker18[:, i] = marker
avg_marker18 = np.mean(marker18, axis=1)
std_marker18 = np.std(marker18, axis=1)
median_marker18 = np.median(marker18, axis=1)
q25_marker18 = np.quantile(marker18, 0.15, axis=1)
q75_marker18 = np.quantile(marker18, 0.85, axis=1)
#
#
#




# %% Figures
# plt.style.use('./stylesheets/prb.mplstyle')
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
# axcolour = ['#FF7F00', '#00C957', '#CD3278', '#6495ED']


# Mean value of the marker
# fig, ax = plt.subplots(figsize=(6, 5))
# #ax.plot(gamma12, avg_marker12, color=axcolour[0], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L12, Nsamples12))
# ax.plot(gamma10, avg_marker10, color=axcolour[1], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L10, Nsamples10))
# ax.plot(gamma14, avg_marker14, color=axcolour[1], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L14, Nsamples14))
# # ax.plot(gamma16, avg_marker16, color=axcolour[2], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L16, Nsamples16))
# # ax.plot(gamma18, avg_marker18, color=axcolour[3], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L18, Nsamples18))
# ax.set_xlim([-8, 8])
# ax.set_ylim([-1.5, 1.5])
# ax.tick_params(which='major', width=0.75, labelsize=20, direction='in', top=True, right=True)
# ax.tick_params(which='major', length=14, labelsize=20, direction='in', top=True, right=True)
# ax.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
# ax.tick_params(which='minor', length=7, direction='in', top=True, right=True)
# majorsy = [-1, 0, 1]
# minorsy = [-1.5, -0.5, 0.5, 1.5]
# majorsx = [-8, -4, 0, 4, 8]
# minorsx = [-6, -2, 2, 6]
# ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
# ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax.set_ylabel("$\overline{\\nu}$", fontsize=20)
# ax.set_xlabel("$\\\delta$", fontsize=20)
# ax.legend(bbox_to_anchor=(0.6, 0.8), loc='upper right', frameon=False, fontsize=13)
# plt.tight_layout()
# plt.savefig("XYZmaj_marker_average.pdf", bbox_inches="tight")
# plt.show()





# Medians
fig1, ax = plt.subplots(figsize=(6, 5))
#ax.plot(gamma12, median_marker12, color=axcolour[0], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L12, Nsamples12))
ax.plot(gamma10, median_marker10, color=axcolour[0], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L10, Nsamples10))
ax.plot(gamma14, median_marker14, color=axcolour[1], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L14, Nsamples14))
#ax.plot(gamma16, median_marker16, color=axcolour[3], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L16, Nsamples16))
ax.plot(gamma18, median_marker18, color=axcolour[3], marker='.', markersize=12, label='$L=$ {}, $N=$ {}'.format(L18, Nsamples18))
ax.fill_between(gamma10, q25_marker10, q75_marker10, color=axcolour[0], alpha=0.3)
ax.fill_between(gamma14, q25_marker14, q75_marker14, color=axcolour[1], alpha=0.3)
#ax.fill_between(gamma16, q25_marker16, q75_marker16, color=axcolour[3], alpha=0.3)
ax.fill_between(gamma18, q25_marker18, q75_marker18, color=axcolour[3], alpha=0.3)

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
ax.set_ylabel("med$(\\nu)$", fontsize=20)
ax.set_xlabel("$\delta$", fontsize=20)
ax.legend(bbox_to_anchor=(0.6, 0.8), loc='center right', frameon=False, fontsize=16)
plt.tight_layout()
plt.savefig("XYZmaj_marker_median.pdf", bbox_inches="tight")
plt.show()

# Median and quantiles
# fig2 = plt.figure(figsize=(8, 8))
# gs = GridSpec(2, 2, figure=fig2, wspace=0.2, hspace=0.2)
# ax1 = fig2.add_subplot(gs[0, 0])
# ax2 = fig2.add_subplot(gs[0, 1])
# ax3 = fig2.add_subplot(gs[1, 0])
# ax4 = fig2.add_subplot(gs[1, 1])
# 
# ax1.plot(gamma12, median_marker12, color=axcolour[0], marker='.', markersize=2, linewidth=0.3, label='$L=$ {}, $N=$ {}'.format(L12, Nsamples12))
# ax1.fill_between(gamma12, q25_marker12, q75_marker12, color=axcolour[0], alpha=0.3)
# ax1.set_xlim([-8, 8])
# ax1.set_ylim([-1.5, 1.5])
# ax1.tick_params(which='major', width=0.75, labelsize=20)
# ax1.tick_params(which='major', length=14, labelsize=20)
# ax1.tick_params(which='minor', width=0.75)
# ax1.tick_params(which='minor', length=7)
# majorsy = [-1, 0, 1]
# minorsy = [-1.5, -0.5, 0.5, 1.5]
# majorsx = [-8, -4, 0, 4, 8]
# minorsx = [-6, -2, 2, 6]
# ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
# ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax1.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax1.set_ylabel("med$(\\nu)$", fontsize=10)
# ax1.legend(loc='upper right', frameon=False, fontsize=10)
# 
# 
# 
# ax2.plot(gamma14, median_marker14, color=axcolour[1], marker='.', markersize=2, linewidth=0.3, label='$L=$ {}, $N=$ {}'.format(L14, Nsamples14))
# ax2.fill_between(gamma14, q25_marker14, q75_marker14, color=axcolour[1], alpha=0.3)
# ax2.set_xlim([-8, 8])
# ax2.set_ylim([-1.5, 1.5])
# ax2.tick_params(which='major', width=0.75, labelsize=20)
# ax2.tick_params(which='major', length=14, labelsize=20)
# ax2.tick_params(which='minor', width=0.75)
# ax2.tick_params(which='minor', length=7)
# majorsy = [-1, 0, 1]
# minorsy = [-1.5, -0.5, 0.5, 1.5]
# majorsx = [-8, -4, 0, 4, 8]
# minorsx = [-6, -2, 2, 6]
# ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
# ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax2.legend(loc='upper right', frameon=False, fontsize=10)
# 
# 
# 
# 
# 
# ax3.plot(gamma16, median_marker16, color=axcolour[2], marker='.', markersize=2, linewidth=0.3, label='$L=$ {}, $N=$ {}'.format(L16, Nsamples16))
# ax3.fill_between(gamma16, q25_marker16, q75_marker16, color=axcolour[2], alpha=0.3)
# ax3.set_xlim([-8, 8])
# ax3.set_ylim([-1.5, 1.5])
# ax3.tick_params(which='major', width=0.75, labelsize=20)
# ax3.tick_params(which='major', length=14, labelsize=20)
# ax3.tick_params(which='minor', width=0.75)
# ax3.tick_params(which='minor', length=7)
# majorsy = [-1, 0, 1]
# minorsy = [-1.5, -0.5, 0.5, 1.5]
# majorsx = [-8, -4, 0, 4, 8]
# minorsx = [-6, -2, 2, 6]
# ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax3.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
# ax3.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax3.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax3.legend(loc='upper right', frameon=False, fontsize=10)
# ax3.set_ylabel("med$(\\nu)$", fontsize=10)
# ax3.set_xlabel("$\delta$", fontsize=10)


#
# ax4.plot(gamma18, median_marker18, color=axcolour[3], marker='.', markersize=2, linewidth=0.3, label='$L=$ {}, $N=$ {}'.format(L18, Nsamples18))
# ax4.fill_between(gamma18, q25_marker18, q75_marker18, color=axcolour[3], alpha=0.3)
# ax4.set_xlim([-8, 8])
# ax4.set_ylim([-1.5, 1.5])
# ax4.tick_params(which='major', width=0.75, labelsize=20)
# ax4.tick_params(which='major', length=14, labelsize=20)
# ax4.tick_params(which='minor', width=0.75)
# ax4.tick_params(which='minor', length=7)
# majorsy = [-1, 0, 1]
# minorsy = [-1.5, -0.5, 0.5, 1.5]
# majorsx = [-8, -4, 0, 4, 8]
# minorsx = [-6, -2, 2, 6]
# ax4.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax4.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
# ax4.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
# ax4.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# ax4.set_xlabel("$\delta$", fontsize=10)
# ax4.legend(loc='upper right', frameon=False, fontsize=10)

# plt.tight_layout()
plt.savefig("XYZmaj_marker_quantiles.pdf", bbox_inches="tight")
plt.show()















