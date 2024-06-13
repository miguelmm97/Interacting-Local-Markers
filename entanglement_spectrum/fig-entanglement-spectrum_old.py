#%% Modules

# Math
import numpy as np
from numpy import pi

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# Handling data
import os
import h5py


#%% Loading data
with h5py.File('PlotData.hdf5', 'r') as f:

    # SPT +1
    # Markers
    marker_median_1 = f['KitaevMarkers/Medians'][()]
    marker_upper75_1 = f['KitaevMarkers/Upper75%Region'][()]
    marker_lower75_1 = f['KitaevMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_1 = f['KitaevSpacing/Medians'][()]
    spacing_upper75_1 = f['KitaevSpacing/Upper75%Region'][()]
    spacing_lower75_1 = f['KitaevSpacing/Lower75%Region'][()]

    # SPT -1
    # Markers
    marker_median_m1 = f['NegativeKitaevMarkers/Medians'][()]
    marker_upper75_m1 = f['NegativeKitaevMarkers/Upper75%Region'][()]
    marker_lower75_m1 = f['NegativeKitaevMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_m1 = f['NegativeKitaevSpacing/Medians'][()]
    spacing_upper75_m1 = f['NegativeKitaevSpacing/Upper75%Region'][()]
    spacing_lower75_m1 = f['NegativeKitaevSpacing/Lower75%Region'][()]


    # Trivial
    # Markers
    marker_median_0 = f['OneLongSingletMarkers/Medians'][()]
    marker_upper75_0 = f['OneLongSingletMarkers/Upper75%Region'][()]
    marker_lower75_0 = f['OneLongSingletMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_0 = f['OneLongSingletSpacing/Medians'][()]
    spacing_upper75_0 = f['OneLongSingletSpacing/Upper75%Region'][()]
    spacing_lower75_0 = f['OneLongSingletSpacing/Lower75%Region'][()]



gates = np.arange(len(marker_median_1))
#%% Figure average markers

# Style, font and colours
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
red, blue, yellow, shadeblue = (0.922526, 0.385626, 0.209179), (0.368417, 0.506779, 0.709798),  (1, 0.75, 0), '#00BFFF'
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C']

# State labels
label_1 = '$|\psi_+ \\rangle $'
label_m1 = '$| \psi_- \\rangle$'
label_0 = '$| \psi_0 \\rangle$'


# Figure medians
fig = plt.figure(figsize=(7, 6))
gs = GridSpec(1, 2, figure=fig, wspace=0.6)
ax1 = fig.add_subplot(gs[0, 0])


# Figure entanglement spectrum

# Medians
spacing_median_1[spacing_median_1 < 1e-16] = 1e-16
spacing_median_m1[spacing_median_m1 < 1e-16] = 1e-16
spacing_median_0[spacing_median_0 < 1e-16] = 1e-16
ax.plot(gates, spacing_median_1, color=axcolour[1], marker='D', markersize=4.5, label=label_1)
ax.plot(gates, spacing_median_m1, color=axcolour[2], marker='^', markersize=4.5, label=label_m1)
ax.plot(gates, spacing_median_0, color=axcolour[3], marker='o', markersize=4.5, label=label_0)
ax.set_yscale('log')

# Filling
spacing_upper75_1[spacing_upper75_1 < 1e-16] = 1e-16
spacing_upper75_0[spacing_upper75_0 < 1e-16] = 1e-16
spacing_upper75_m1[spacing_upper75_m1 < 1e-16] = 1e-16

spacing_lower75_1[spacing_lower75_1 < 1e-16] = 1e-16
spacing_lower75_0[spacing_lower75_0 < 1e-16] = 1e-16
spacing_lower75_m1[spacing_lower75_m1 < 1e-16] = 1e-16

ax.fill_between(gates, spacing_lower75_1,  spacing_upper75_1, color=axcolour[1], alpha=0.15)
ax.fill_between(gates, spacing_lower75_m1, spacing_upper75_m1, color=axcolour[2], alpha=0.15)
ax.fill_between(gates, spacing_lower75_0,  spacing_upper75_0, color=shadeblue, alpha=0.15)

# Legend, labels and text
# ax.text(9, 0.1e-1, '$p=0.25$', fontsize=15)
ax.legend(bbox_to_anchor=[0.7, 0.95], ncol=1, frameon=False, fontsize=15)

# Axis limits and labels
ax.set_ylim([1e-18, 1])
ax.set_xlim([0, 14])
ax.set_ylabel("$\delta$", fontsize=20)
ax.set_xlabel("Circuit depth", fontsize=20)

# Tick params
ax.tick_params(which='major', width=0.75, labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='major', length=5,  labelsize=15, direction='in', top=True, right=True)
ax.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax.tick_params(which='minor', length=2.5, direction='in', top=True, right=True)

# majors and minors
majorsy = [1e-16, 1e-12, 1e-8, 1e-4, 1]
minorsy = [1e-14, 1e-10, 1e-6, 1e-2]
majorsx = [0, 2, 4, 6, 8, 10, 12, 14]
minorsx = [1, 3, 5, 7, 9, 11, 13]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
minorsy_str = []
ax.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))



left, bottom, width, height = [0.45, 0.15, 0.5, 0.5]
inset_ax = ax.inset_axes((left, bottom, width, height))

# Medians
inset_ax.plot(gates, marker_median_1, color=axcolour[1], marker='D', markersize=4.5, label=label_1)
inset_ax.plot(gates, marker_median_m1, color=axcolour[2], marker='^', markersize=4.5, label=label_m1)
inset_ax.plot(gates, marker_median_0, color=axcolour[3], marker='o', markersize=4.5, label=label_0)

# Legend labels and text
# inset_ax.text(0.75, 0.8, label_1, fontsize=16)
# inset_ax.text(0.75, -0.9, label_m1, fontsize=16)
# inset_ax.text(0.75, 0.1, label_0, fontsize=16)
# inset_ax.text(3, -0.5, '$p=0.25$', fontsize=16)

# Filling
inset_ax.fill_between(gates, marker_lower75_1,  marker_upper75_1, color=axcolour[1], alpha=0.15)
inset_ax.fill_between(gates, marker_lower75_m1, marker_upper75_m1, color=axcolour[2], alpha=0.15)
inset_ax.fill_between(gates, marker_lower75_0,  marker_upper75_0, color=shadeblue, alpha=0.15)

# Axis limits and labels
inset_ax.set_ylim([-1.25, 1.25])
inset_ax.set_xlim([0, gates[-1]])
inset_ax.set_ylabel("$\\nu$", fontsize=15)
inset_ax.set_xlabel("Circuit depth", fontsize=15)

# Tick params
inset_ax.tick_params(which='major', width=0.75, labelsize=15, direction='in', top=True, right=True)
inset_ax.tick_params(which='major', length=5,  labelsize=15, direction='in', top=True, right=True)
inset_ax.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax.tick_params(which='minor', length=2.5, direction='in', top=True, right=True)

# majors and minors
majorsy = [-1, 0, 1]
minorsy = [-0.5, 0.5]
majorsx = [0, 4, 8, 12]
minorsx = [2, 6, 10, 14]
inset_ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
inset_ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
inset_ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
inset_ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))




plt.savefig("entanglement_spectrum1.pdf")
plt.show()