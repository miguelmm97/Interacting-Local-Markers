import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import h5py


# %% Reading system size 12
with h5py.File('KiwiPlot.hdf5', 'r') as f:

    # System size 5
    # median5 = f['l=5/Medians'][()]
    # delta5 = f['l=5/DeltaValues'][()]
    # qtop5 = f['l=5/Upper75%Region'][()]
    # qbottom5 = f['l=5/Lower75%Region'][()]

    # System size 9
    median9 = f['l=9/Medians'][()]
    delta9 = f['l=9/DeltaValues'][()]
    qtop9 = f['l=9/Upper75%Region'][()]
    qbottom9 = f['l=9/Lower75%Region'][()]

    # System size 13
    median13 = f['l=13/Medians'][()]
    delta13 = f['l=13/DeltaValues'][()]
    qtop13 = f['l=13/Upper75%Region'][()]
    qbottom13 = f['l=13/Lower75%Region'][()]

    # System size 17
    median17 = f['l=17/Medians'][()]
    delta17 = f['l=17/DeltaValues'][()]
    qtop17 = f['l=17/Upper75%Region'][()]
    qbottom17 = f['l=17/Lower75%Region'][()]
    errorup17 = f['l=17/Median95%ConfidenceIntervallUpper'][()]
    errorbottom17 = f['l=17/Median95%ConfidenceIntervallLower'][()]

# indices5 = np.argsort(delta5)
# delta5.sort()
# median5 = median5[indices5]
# qtop5 = qtop5[indices5]
# qbottom5 = qbottom5[indices5]

indices9 = np.argsort(delta9)
delta9.sort()
median9 = median9[indices9]
qtop9 = qtop9[indices9]
qbottom9 = qbottom9[indices9]

indices13 = np.argsort(delta13)
delta13.sort()
median13 = median13[indices13]
qtop13 = qtop13[indices13]
qbottom13 = qbottom13[indices13]

indices17 = np.argsort(delta17)
delta17.sort()
median17 = median17[indices17]
qtop17 = qtop17[indices17]
qbottom17 = qbottom17[indices17]
errorup17 = errorup17[indices17]
errorbottom17 = errorbottom17[indices17]
errorbars17 = np.array([errorup17, errorbottom17])
print(errorup17)
print(errorbottom17)

#%% Figures
# plt.style.use('./stylesheets/prb.mplstyle')
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
# axcolour = ['#FF7F00', '#00C957', '#CD3278', '#6495ED']
red= (0.922526, 0.385626, 0.209179)
blue = (0.368417, 0.506779, 0.709798)
yellow = (1, 0.75, 0)
shadeblue = '#00BFFF'
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C']

# Medians
fig1, ax = plt.subplots(figsize=(6, 5))
# ax.plot(delta5, median5, color=axcolour[0], marker='s', markersize=4.5, label='{}'.format(5))
ax.plot(delta9, median9, color=axcolour[1], marker='D', markersize=4.5, label='{}'.format(9))
ax.plot(delta13, median13, color=axcolour[2], marker='^', markersize=4.5, label='{}'.format(13))
ax.plot(delta17, median17, color=axcolour[3], marker='o', markersize=4.5, label='{}'.format(17))
# ax.fill_between(delta5, qbottom5, qtop5, color=axcolour[0], alpha=0.3)
ax.fill_between(delta9, qbottom9, qtop9, color=axcolour[1], alpha=0.3)
ax.fill_between(delta13, qbottom13, qtop13, color=axcolour[2], alpha=0.3)
ax.fill_between(delta17, qbottom17, qtop17, color=shadeblue, alpha=0.3)
ax.errorbar(delta17, median17, yerr=errorbars17, color=axcolour[3])
ax.set_xlim([-8, 8])
ax.set_ylim([-1, 1.5])
ax.tick_params(which='major', width=0.75, labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='major', length=14,  labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax.tick_params(which='minor', length=7, direction='in', top=True, right=True)
majorsy = [-1, -0.5, 0, 0.5, 1, 1.5]
minorsy = [-0.75, -0.25, 0.25, 0.75, 1.25]
majorsx = [-8, -4, 0, 4, 8]
minorsx = [-6, -2, 2, 6]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
ax.set_ylabel("$\\tilde \\nu$", fontsize=20)
ax.set_xlabel("$\delta$", fontsize=20)
ax.text(-4.75, -0.2, '$L$', fontsize=16)
ax.legend(bbox_to_anchor=(0.29, 0.32), ncol=1, frameon=False, fontsize=16) # bbox_to_anchor=(0.98, 0.6),



left, bottom, width, height = [0.11, 0.65, 0.28, 0.28]
inset_ax1 = ax.inset_axes([left, bottom, width, height])
inset_ax1.plot(delta9, median9, color=axcolour[1], marker='s', markersize=3, label='{}'.format(9))
inset_ax1.plot(delta13, median13, color=axcolour[2], marker='D', markersize=3, label='{}'.format(13))
inset_ax1.plot(delta17, median17, color=axcolour[3], marker='o', markersize=3, label='{}'.format(17))
inset_ax1.fill_between(delta9, qbottom9, qtop9, color=axcolour[1], alpha=0.3)
inset_ax1.fill_between(delta13, qbottom13, qtop13, color=axcolour[2], alpha=0.3)
inset_ax1.fill_between(delta17, qbottom17, qtop17, color=shadeblue, alpha=0.3)
# inset_ax1.set_ylabel("med$(\\nu)$", fontsize=10)
# inset_ax1.set_xlabel("$\delta$", fontsize=10)
inset_ax1.set_xlim([-5, -3])
inset_ax1.set_ylim([-0.02, 0.02])
inset_ax1.tick_params(which='major', width=0.75, labelsize=10, direction='in', top=True, right=True)
inset_ax1.tick_params(which='major', length=7,  labelsize=10, direction='in', top=True, right=True)
inset_ax1.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax1.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
majorsy = [-0.02, 0, 0.02]
minorsy = [-0.01, 0.01]
majorsx = [-5, -4, -3]
minorsx = [-4.5, -3.5]
inset_ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
inset_ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
inset_ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
inset_ax1.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))



left, bottom, width, height = [0.67, 0.10, 0.28, 0.28]
inset_ax2 = ax.inset_axes([left, bottom, width, height])
inset_ax2.plot(delta9, median9, color=axcolour[1], marker='s', markersize=3, label='{}'.format(9))
inset_ax2.plot(delta13, median13, color=axcolour[2], marker='D', markersize=3, label='{}'.format(13))
inset_ax2.plot(delta17, median17, color=axcolour[3], marker='o', markersize=3, label='{}'.format(17))
inset_ax2.fill_between(delta9, qbottom9, qtop9, color=axcolour[1], alpha=0.3)
inset_ax2.fill_between(delta13, qbottom13, qtop13, color=axcolour[2], alpha=0.3)
inset_ax2.fill_between(delta17, qbottom17, qtop17, color=shadeblue, alpha=0.3)
# inset_ax2.set_ylabel("med$(\\nu)$", fontsize=10)
# inset_ax2.set_xlabel("$\delta$", fontsize=10)
inset_ax2.set_xlim([3, 6])
inset_ax2.set_ylim([0.85, 1.02])
inset_ax2.tick_params(which='major', width=0.75, labelsize=10, direction='in', top=True, right=True)
inset_ax2.tick_params(which='major', length=7,  labelsize=10, direction='in', top=True, right=True)
inset_ax2.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax2.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
majorsy = [0.85, 0.9, 0.95, 1]
minorsy = [0.875, 0.925, 0.975]
majorsx = [3, 4, 5, 6]
minorsx = [3.5, 4.5, 5.5]
inset_ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
inset_ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
inset_ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
inset_ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))



plt.tight_layout()
plt.savefig("KiwiPlot.pdf", bbox_inches="tight")
plt.show()
