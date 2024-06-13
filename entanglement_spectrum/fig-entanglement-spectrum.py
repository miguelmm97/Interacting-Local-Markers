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
with h5py.File('PlotData_new.hdf5', 'r') as f:

    # SPT +1
    # Markers
    marker_median_1 = f['KitaevMarkers/Medians'][()]
    marker_upper75_1 = f['KitaevMarkers/Upper75%Region'][()]
    marker_lower75_1 = f['KitaevMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_1 = f['KitaevSpacing/Medians'][()]
    spacing_upper75_1 = f['KitaevSpacing/Upper75%Region'][()]
    spacing_lower75_1 = f['KitaevSpacing/Lower75%Region'][()]

    # Gaps
    gaps_median_1 = f['KitaevGap/Medians'][()]
    gaps_upper75_1 = f['KitaevGap/Upper75%Region'][()]
    gaps_lower75_1 = f['KitaevGap/Lower75%Region'][()]

    # SPT -1
    # Markers
    marker_median_m1 = f['NegativeKitaevMarkers/Medians'][()]
    marker_upper75_m1 = f['NegativeKitaevMarkers/Upper75%Region'][()]
    marker_lower75_m1 = f['NegativeKitaevMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_m1 = f['NegativeKitaevSpacing/Medians'][()]
    spacing_upper75_m1 = f['NegativeKitaevSpacing/Upper75%Region'][()]
    spacing_lower75_m1 = f['NegativeKitaevSpacing/Lower75%Region'][()]

    # Gaps
    gaps_median_m1 = f['NegativeKitaevGap/Medians'][()]
    gaps_upper75_m1 = f['NegativeKitaevGap/Upper75%Region'][()]
    gaps_lower75_m1 = f['NegativeKitaevGap/Lower75%Region'][()]


    # Trivial
    # Markers
    marker_median_0 = f['OneLongSingletMarkers/Medians'][()]
    marker_upper75_0 = f['OneLongSingletMarkers/Upper75%Region'][()]
    marker_lower75_0 = f['OneLongSingletMarkers/Lower75%Region'][()]

    # Spacings
    spacing_median_0 = f['OneLongSingletSpacing/Medians'][()]
    spacing_upper75_0 = f['OneLongSingletSpacing/Upper75%Region'][()]
    spacing_lower75_0 = f['OneLongSingletSpacing/Lower75%Region'][()]

    # Gaps
    gaps_median_0 = f['OneLongSingletGap/Medians'][()]
    gaps_upper75_0 = f['OneLongSingletGap/Upper75%Region'][()]
    gaps_lower75_0 = f['OneLongSingletGap/Lower75%Region'][()]


gates = np.arange(len(marker_median_1))
#%% Figure average markers

# Style, font and colours
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
red, blue, yellow, shadeblue = (0.922526, 0.385626, 0.209179), (0.368417, 0.506779, 0.709798),  (1, 0.75, 0), '#00BFFF'
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C']

# State labels
label_1 = '$|\psi_+ \\rangle $'
label_m1 = '$| \psi_- \\rangle$'
label_0 = '$| \psi_0 \\rangle$'


fig = plt.figure(figsize=(8, 6))
gs = GridSpec(4, 4, figure=fig, wspace=1, hspace=0.2)
ax1 = fig.add_subplot(gs[:, 0:2])
ax2 = fig.add_subplot(gs[0:2, 2:])
ax3 = fig.add_subplot(gs[2:, 2:])


# Markers

# Medians
ax1.plot(gates, marker_median_1, color=axcolour[1], marker='D', markersize=4.5, label=label_1)
ax1.plot(gates, marker_median_m1, color=axcolour[2], marker='^', markersize=4.5, label=label_m1)
ax1.plot(gates, marker_median_0, color=axcolour[3], marker='o', markersize=4.5, label=label_0)
# Legend labels and text
# ax1.text(0.75, 0.8, label_1, fontsize=16)
# ax1.text(0.75, -0.9, label_m1, fontsize=16)
# ax1.text(0.75, 0.1, label_0, fontsize=16)
# ax1.text(3, -0.5, '$p=0.25$', fontsize=16)
# Filling
ax1.fill_between(gates, marker_lower75_1,  marker_upper75_1, color=axcolour[1], alpha=0.15)
ax1.fill_between(gates, marker_lower75_m1, marker_upper75_m1, color=axcolour[2], alpha=0.15)
ax1.fill_between(gates, marker_lower75_0,  marker_upper75_0, color=shadeblue, alpha=0.15)
# Axis limits and labels
ax1.set_ylim([-1.25, 1.25])
ax1.set_xlim([0, gates[-1]])
ax1.set_ylabel("$\\nu$", fontsize=20)
ax1.set_xlabel("Circuit depth", fontsize=20)
# Tick params
ax1.tick_params(which='major', width=0.75, labelsize=20, direction='in', top=True, right=True)
ax1.tick_params(which='major', length=5,  labelsize=20, direction='in', top=True, right=True)
ax1.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax1.tick_params(which='minor', length=2.5, direction='in', top=True, right=True)
# majors and minors
majorsy = [-1, 0, 1]
minorsy = [-0.5, 0.5]
majorsx = [0, 4, 8, 12]
minorsx = [2, 6, 10, 14]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax1.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))




# Entanglement spectrum

# Medians
spacing_median_1[spacing_median_1 < 1e-16] = 1e-16
spacing_median_m1[spacing_median_m1 < 1e-16] = 1e-16
spacing_median_0[spacing_median_0 < 1e-16] = 1e-16
ax2.plot(gates, spacing_median_1, color=axcolour[1], marker='D', markersize=4.5, label=label_1)
ax2.plot(gates, spacing_median_m1, color=axcolour[2], marker='^', markersize=4.5, label=label_m1)
ax2.plot(gates, spacing_median_0, color=axcolour[3], marker='o', markersize=4.5, label=label_0)
# ax2.set_yscale('log')
# Filling
spacing_upper75_1[spacing_upper75_1 < 1e-16] = 1e-16
spacing_upper75_0[spacing_upper75_0 < 1e-16] = 1e-16
spacing_upper75_m1[spacing_upper75_m1 < 1e-16] = 1e-16
spacing_lower75_1[spacing_lower75_1 < 1e-16] = 1e-16
spacing_lower75_0[spacing_lower75_0 < 1e-16] = 1e-16
spacing_lower75_m1[spacing_lower75_m1 < 1e-16] = 1e-16
ax2.fill_between(gates, spacing_lower75_1,  spacing_upper75_1, color=axcolour[1], alpha=0.15)
ax2.fill_between(gates, spacing_lower75_m1, spacing_upper75_m1, color=axcolour[2], alpha=0.15)
ax2.fill_between(gates, spacing_lower75_0,  spacing_upper75_0, color=shadeblue, alpha=0.15)
# Legend, labels and text
# ax2.text(9, 0.1e-1, '$p=0.25$', fontsize=15)
# ax2.legend(bbox_to_anchor=[0.45, 0.95], ncol=1, frameon=False, fontsize=15)
# ax2is limits and labels
ax2.set_ylim([-0.03, 0.6])
ax2.set_xlim([0, gates[-1]])
ax2.set_ylabel("$\delta$", fontsize=20)
# ax2.set_xlabel("Circuit depth", fontsize=20)
# Tick params
ax2.tick_params(which='major', width=0.75, labelsize=20, direction='in', top=True, right=True)
ax2.tick_params(which='major', length=5,  labelsize=20, direction='in', top=True, right=True)
ax2.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax2.tick_params(which='minor', length=2.5, direction='in', top=True, right=True)
# majors and minors
# majorsy = [1e-16, 1e-12, 1e-8, 1e-4, 1]
majorsy = [0, 0.2, 0.4, 0.6]
# minorsy = [1e-14, 1e-10, 1e-6, 1e-2]
minorsy = [0.1, 0.3, 0.5]
minorsy = [0.1, 0.3, 0.5]
majorsx = [0, 4, 8, 12]
ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
minorsy_str = []
majorsx_str = []
ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))
ax2.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))



# Gaps

# Medians
# gaps_median_1[spacing_median_1 < 1e-16] = 1e-16
# gaps_median_m1[spacing_median_m1 < 1e-16] = 1e-16
# gaps_median_0[spacing_median_0 < 1e-16] = 1e-16
ax3.plot(gates, gaps_median_1, color=axcolour[1], marker='D', markersize=4.5, label=label_1)
ax3.plot(gates, gaps_median_m1, color=axcolour[2], marker='^', markersize=4.5, label=label_m1)
ax3.plot(gates, gaps_median_0, color=axcolour[3], marker='o', markersize=4.5, label=label_0)
# ax3.set_yscale('log')
# Filling
# gaps_upper75_1[spacing_upper75_1 < 1e-16] = 1e-16
# gaps_upper75_0[spacing_upper75_0 < 1e-16] = 1e-16
# gaps_upper75_m1[spacing_upper75_m1 < 1e-16] = 1e-16
# spacing_lower75_1[spacing_lower75_1 < 1e-16] = 1e-16
# spacing_lower75_0[spacing_lower75_0 < 1e-16] = 1e-16
# spacing_lower75_m1[spacing_lower75_m1 < 1e-16] = 1e-16
ax3.fill_between(gates, gaps_lower75_1,  gaps_upper75_1, color=axcolour[1], alpha=0.15)
ax3.fill_between(gates, gaps_lower75_m1, gaps_upper75_m1, color=axcolour[2], alpha=0.15)
ax3.fill_between(gates, gaps_lower75_0,  gaps_upper75_0, color=shadeblue, alpha=0.15)

# Legend, labels and text
# ax2.text(9, 0.1e-1, '$p=0.25$', fontsize=15)
# ax3.legend(bbox_to_anchor=[0.7, 0.95], ncol=1, frameon=False, fontsize=15)

# axis limits and labels
ax3.set_ylim([0, 1])
ax3.set_xlim([0, gates[-1]])
ax3.set_ylabel("$\Delta$", fontsize=20)
ax3.set_xlabel("Circuit depth", fontsize=20)
ax3.legend(bbox_to_anchor=[0.35, 0.85], ncol=1, frameon=False, fontsize=20)

# Tick params
ax3.tick_params(which='major', width=0.75, labelsize=15, direction='in', top=True, right=True)
ax3.tick_params(which='major', length=5,  labelsize=15, direction='in', top=True, right=True)
ax3.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax3.tick_params(which='minor', length=2.5, direction='in', top=True, right=True)

# majors and minors
majorsy = [0, 0.2, 0.4, 0.6, 0.8, 1]
minorsy = [0.1, 0.3, 0.5, 0.7, 0.9]
majorsx = [0, 4, 8, 12]
minorsx = [2, 6, 10]
ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax3.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax3.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax3.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
# minorsy_str = []
# ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))







plt.savefig("entanglement_spectrum1.pdf")
plt.show()