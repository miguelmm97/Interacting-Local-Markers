import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import os
import h5py


# %% Data
with h5py.File('ProbDist.hdf5', 'r') as f:
    data = f['ProbabilityDistributionFunctions'][()]
    data_inset_marker = f['NuWidthOf75%OfData'][()]
    data_inset_prob = f['MaximumProbabilityDensities'][()]


# Delta 1
marker5_delta1 = data[0, 0, :, 0]
prob5_delta1 = data[0, 0, :, 1]
marker9_delta1 = data[0, 1, :, 0]
prob9_delta1 = data[0, 1, :, 1]
marker13_delta1 = data[0, 2, :, 0]
prob13_delta1 = data[0, 2, :, 1]
marker17_delta1 = data[0, 3, :, 0]
prob17_delta1 = data[0, 3, :, 1]
inset_markerX_delta1 = data_inset_marker[0, :, 0, 0]
inset_markerY_delta1 = data_inset_marker[0, :, 0, 1]
inset_probX_delta1 = data_inset_prob[0, :, 0, 0]
inset_probY_delta1 = data_inset_prob[0, :, 0, 1]
y = np.polyfit(inset_markerX_delta1[2:], np.log(inset_markerY_delta1[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_marker1 = a * np.exp(b * x)
y = np.polyfit(inset_probX_delta1[2:], np.log(inset_probY_delta1[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_prob1 = a * np.exp(b * x)

# Delta 2
marker5_delta2 = data[1, 0, :, 0]
prob5_delta2 = data[1, 0, :, 1]
marker9_delta2 = data[1, 1, :, 0]
prob9_delta2 = data[1, 1, :, 1]
marker13_delta2 = data[1, 2, :, 0]
prob13_delta2 = data[1, 2, :, 1]
marker17_delta2 = data[1, 3, :, 0]
prob17_delta2 = data[1, 3, :, 1]
inset_markerX_delta2 = data_inset_marker[1, :, 0, 0]
inset_markerY_delta2 = data_inset_marker[1, :, 0, 1]
inset_probX_delta2 = data_inset_prob[1, :, 0, 0]
inset_probY_delta2 = data_inset_prob[1, :, 0, 1]
y = np.polyfit(inset_markerX_delta2[2:], np.log(inset_markerY_delta2[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_marker2 = a * np.exp(b * x)
y = np.polyfit(inset_probX_delta2[2:], np.log(inset_probY_delta2[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_prob2 = a * np.exp(b * x)


# Delta 3
marker5_delta3 = data[2, 0, :, 0]
prob5_delta3 = data[2, 0, :, 1]
marker9_delta3 = data[2, 1, :, 0]
prob9_delta3 = data[2, 1, :, 1]
marker13_delta3 = data[2, 2, :, 0]
prob13_delta3 = data[2, 2, :, 1]
marker17_delta3 = data[2, 3, :, 0]
prob17_delta3 = data[2, 3, :, 1]
inset_markerX_delta3 = data_inset_marker[2, :, 0, 0]
inset_markerY_delta3 = data_inset_marker[2, :, 0, 1]
inset_probX_delta3 = data_inset_prob[2, :, 0, 0]
inset_probY_delta3 = data_inset_prob[2, :, 0, 1]
y = np.polyfit(inset_markerX_delta3[2:], np.log(inset_markerY_delta3[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_marker3 = a * np.exp(b * x)
y = np.polyfit(inset_probX_delta3[2:], np.log(inset_probY_delta3[2:]), 1)
a = np.exp(y[1]); b = y[0]; x = np.linspace(4, 18, 20)
exp_prob3 = a * np.exp(b * x)

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 28, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF', '#6495ED', '#DC143C']
red= (0.922526, 0.385626, 0.209179)
blue = (0.368417, 0.506779, 0.709798)
yellow = (1, 0.75, 0)
axcolour = [yellow, '#FF416D', 	'#D15FEE', 	'#1C86EE', '#6495ED', '#DC143C']

fig = plt.figure(figsize=(27, 8))
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


# Delta1
ax1.plot(marker5_delta1, prob5_delta1, color=axcolour[0], marker='s', markersize=6, label='{}'.format(5))
ax1.plot(marker9_delta1, prob9_delta1, color=axcolour[1], marker='D', markersize=6, label='{}'.format(9))
ax1.plot(marker13_delta1, prob13_delta1, color=axcolour[2], marker='^', markersize=6, label='{}'.format(13))
ax1.plot(marker17_delta1, prob17_delta1, color=axcolour[3], marker='o', markersize=6, label='{}'.format(17))
ax1.set_yscale('log')
ax1.set_xlim([-0.5, 1.1])
ax1.set_ylim([0.1, 100])
ax1.tick_params(which='major', width=0.75, labelsize=28, direction='in', top=True, right=True)
ax1.tick_params(which='major', length=14,  labelsize=28, direction='in', top=True, right=True)
ax1.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax1.tick_params(which='minor', length=7, direction='in', top=True, right=True)
ax1.set_xlabel("$\\nu$", fontsize=33)
ax1.set_ylabel("$P(\\nu)$", fontsize=33)
ax1.text(0.85, 70, '$L$', fontsize=23)
ax1.text(0.7, 6.8, '$\delta=2.8$', fontsize=23)
ax1.legend(bbox_to_anchor=(0.7, 0.95), loc='upper left', ncol=1, frameon=False, fontsize=23)  # bbox_to_anchor=(0.98, 0.6)
ax1.text(-0.438, 67.7, '$(a)$', fontsize=23)

left, bottom, width, height = [0.15, 0.5, 0.4, 0.4]
inset_ax1 = ax1.inset_axes([left, bottom, width, height])
inset_ax1.plot(inset_markerX_delta1[0], inset_markerY_delta1[0], marker='o', color=axcolour[4], markersize=6, label='$\Delta \\nu$')
inset_ax1.plot(inset_markerX_delta1[1], inset_markerY_delta1[1], marker='o', color=axcolour[4], markersize=6)
inset_ax1.plot(inset_markerX_delta1[2], inset_markerY_delta1[2], marker='o', color=axcolour[4], markersize=6)
inset_ax1.plot(inset_markerX_delta1[3], inset_markerY_delta1[3], marker='o', color=axcolour[4], markersize=6)
inset_ax1.plot(np.linspace(4, 18, 20), exp_marker1, color=axcolour[4])
inset_ax1.plot()
inset_ax1.set_ylabel("$\Delta \\nu$", fontsize=23)
inset_ax1.set_xlabel("$L$", fontsize=23)
inset_ax1.yaxis.set_label_coords(-0.075, 0.5)
inset_ax1.set_xlim([4, 18])
inset_ax1.set_ylim([0.1, 1])
inset_ax1.set_yscale('log')
inset_ax1.tick_params(which='major', width=0.75, labelsize=23, direction='in', top=True, right=True)
inset_ax1.tick_params(which='major', length=7,  labelsize=23, direction='in', top=True, right=True)
inset_ax1.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax1.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
minorsy = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
minorsy_str = ['', '', '', '', '', '', '', '']
majorsx = [5, 7, 9, 11, 13, 15, 17]
inset_ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))
inset_ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
inset_ax1.xaxis.set_major_locator(ticker.FixedLocator(majorsx))

right_ax1 = inset_ax1.twinx()
right_ax1.plot(inset_probX_delta1[0], inset_probY_delta1[0], marker='s', color=axcolour[5], markersize=6,label='$P_{\\rm{max}}$')
right_ax1.plot(inset_probX_delta1[1], inset_probY_delta1[1], marker='s', color=axcolour[5], markersize=6)
right_ax1.plot(inset_probX_delta1[2], inset_probY_delta1[2], marker='s', color=axcolour[5], markersize=6)
right_ax1.plot(inset_probX_delta1[3], inset_probY_delta1[3], marker='s', color=axcolour[5], markersize=6)
right_ax1.plot(np.linspace(4, 18, 20), exp_prob1, color=axcolour[5])
right_ax1.set_ylabel("$P_{\\rm{max}}$", fontsize=23)
right_ax1.yaxis.set_label_coords(1.075, 0.5)
right_ax1.set_ylim([1, 10])
right_ax1.set_yscale('log')
right_ax1.tick_params(which='major', width=0.75, labelsize=23, direction='in', top=True, right=True)
right_ax1.tick_params(which='major', length=7,  labelsize=23, direction='in', top=True, right=True)
right_ax1.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
right_ax1.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
minorsy = [1, 2, 3, 4, 5, 6, 7, 8]
minorsy_str = ["", "", "", "", "", "", "", ""]
# right_ax1.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy_str))
# right_ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# minorsx = []
# ax3.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))
right_ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
right_ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))
# right_ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))

inset_ax1.legend(bbox_to_anchor=(-0.07, 1.25), loc='upper left', ncol=1, frameon=False, fontsize=22)  # bbox_to_anchor=(0.98, 0.6)
right_ax1.legend(bbox_to_anchor=(0.38, 1.25), loc='upper left', ncol=1, frameon=False, fontsize=22)  # bbox_to_anchor=(0.98, 0.6)



# Delta2
ax2.plot(marker5_delta2, prob5_delta2, color=axcolour[0], marker='s', markersize=6, label='{}'.format(5))
ax2.plot(marker9_delta2, prob9_delta2, color=axcolour[1], marker='D', markersize=6, label='{}'.format(9))
ax2.plot(marker13_delta2, prob13_delta2, color=axcolour[2], marker='^', markersize=6, label='{}'.format(13))
ax2.plot(marker17_delta2, prob17_delta2, color=axcolour[3], marker='o', markersize=6, label='{}'.format(17))
ax2.set_yscale('log')
ax2.set_xlabel("$\\nu$", fontsize=33)
ax2.set_xlim([0, 1.1])
ax2.set_ylim([0.1, 1000])
ax2.tick_params(which='major', width=0.75, labelsize=28, direction='in', top=True, right=True)
ax2.tick_params(which='major', length=14,  labelsize=28, direction='in', top=True, right=True)
ax2.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax2.tick_params(which='minor', length=7, direction='in', top=True, right=True)
# ax2.text(0.87, 500, '$L$', fontsize=20)
ax2.text(0.8, 30, '$\delta=3.6$', fontsize=23)
# ax2.legend(bbox_to_anchor=(0.7, 0.92), loc='upper left', ncol=1, frameon=False, fontsize=16)  # bbox_to_anchor=(0.98, 0.6)
ax2.text(0.037, 590, '$(b)$', fontsize=23)

left, bottom, width, height = [0.15, 0.5, 0.4, 0.4]
inset_ax2 = ax2.inset_axes([left, bottom, width, height])
inset_ax2.plot(inset_markerX_delta2[0], inset_markerY_delta2[0], marker='o', color=axcolour[4], markersize=6, label='max$(\\nu)$')
inset_ax2.plot(inset_markerX_delta2[1], inset_markerY_delta2[1], marker='o', color=axcolour[4], markersize=6)
inset_ax2.plot(inset_markerX_delta2[2], inset_markerY_delta2[2], marker='o', color=axcolour[4], markersize=6)
inset_ax2.plot(inset_markerX_delta2[3], inset_markerY_delta2[3], marker='o', color=axcolour[4], markersize=6)
inset_ax2.plot(np.linspace(4, 18, 20), exp_marker2, color=axcolour[4])
inset_ax2.set_ylabel("$\Delta \\nu$", fontsize=23)
inset_ax2.set_xlabel("$L$", fontsize=23)
inset_ax2.set_xlim([4, 18])
inset_ax2.set_ylim([0.03, 1])
inset_ax2.set_yscale('log')
inset_ax2.tick_params(which='major', width=0.75, labelsize=23, direction='in', top=True, right=True)
inset_ax2.tick_params(which='major', length=7,  labelsize=23, direction='in', top=True, right=True)
inset_ax2.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax2.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
# majorsy = [0.05, 0.1, 0.20, 0.50, 1]
# majorsy_str = ["$0.05$", "$0.1$", "$0.2$", "$0.5$", "$1$"]
# inset_ax2.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy_str))
# inset_ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
majorsx = [5, 7, 9, 11, 13, 15, 17]
# ax3.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))
inset_ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))

right_ax2 = inset_ax2.twinx()
right_ax2.plot(inset_probX_delta2[0], inset_probY_delta2[0], marker='s', color=axcolour[5], markersize=6,label='$P_{\\rm{max}}$')
right_ax2.plot(inset_probX_delta2[1], inset_probY_delta2[1], marker='s', color=axcolour[5], markersize=6)
right_ax2.plot(inset_probX_delta2[2], inset_probY_delta2[2], marker='s', color=axcolour[5], markersize=6)
right_ax2.plot(inset_probX_delta2[3], inset_probY_delta2[3], marker='s', color=axcolour[5], markersize=6)
right_ax2.plot(np.linspace(4, 18, 20), exp_prob2, color=axcolour[5])
right_ax2.set_ylabel('$P_{\\rm{max}}$', fontsize=23)
right_ax2.set_ylim([1, 1000])
right_ax2.set_yscale('log')
right_ax2.tick_params(which='major', width=0.75, labelsize=23, direction='in', top=True, right=True)
right_ax2.tick_params(which='major', length=7,  labelsize=23, direction='in', top=True, right=True)
right_ax2.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
right_ax2.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
majorsy = [5, 10, 50, 100, 150]
majorsy_str = ["$5$", "$10$", "$50$", "$100$", "$150$"]

# right_ax2.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy_str))
# right_ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))

# inset_ax2.legend(bbox_to_anchor=(-0.12, 1.2), loc='upper left', ncol=1, frameon=False, fontsize=20)  # bbox_to_anchor=(0.98, 0.6)
# right_ax2.legend(bbox_to_anchor=(0.45, 1.2), loc='upper left', ncol=1, frameon=False, fontsize=20)  # bbox_to_anchor=(0.98, 0.6)



# Delta3
ax3.plot(marker5_delta3, prob5_delta3, color=axcolour[0], marker='s', markersize=6, label='{}'.format(5))
ax3.plot(marker9_delta3, prob9_delta3, color=axcolour[1], marker='D', markersize=6, label='{}'.format(9))
ax3.plot(marker13_delta3, prob13_delta3, color=axcolour[2], marker='^', markersize=6, label='{}'.format(13))
ax3.plot(marker17_delta3, prob17_delta3, color=axcolour[3], marker='o', markersize=6, label='{}'.format(17))
ax3.set_xlabel("$\\nu$", fontsize=33)
ax3.set_yscale('log')
ax3.set_xlim([0, 1.1])
ax3.set_ylim([0.1, 10000])
ax3.tick_params(which='major', width=0.75, labelsize=28, direction='in', top=True, right=True)
ax3.tick_params(which='major', length=14,  labelsize=28, direction='in', top=True, right=True)
ax3.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
ax3.tick_params(which='minor', length=7, direction='in', top=True, right=True)
# ax3.text(0.87, 4000, '$L$', fontsize=20)
ax3.text(0.8, 120, '$\delta=4.4$', fontsize=23)
# ax3.legend(bbox_to_anchor=(0.7, 0.92), loc='upper left', ncol=1, frameon=False, fontsize=16)  # bbox_to_anchor=(0.98, 0.6)
ax3.text(0.037, 5200, '$(c)$', fontsize=23)

left, bottom, width, height = [0.15, 0.5, 0.4, 0.4]
inset_ax3 = ax3.inset_axes([left, bottom, width, height])
inset_ax3.plot(inset_markerX_delta3[0], inset_markerY_delta3[0], marker='o', color=axcolour[4], markersize=6, label='max$(\\nu)$')
inset_ax3.plot(inset_markerX_delta3[1], inset_markerY_delta3[1], marker='o', color=axcolour[4], markersize=6)
inset_ax3.plot(inset_markerX_delta3[2], inset_markerY_delta3[2], marker='o', color=axcolour[4], markersize=6)
inset_ax3.plot(inset_markerX_delta3[3], inset_markerY_delta3[3], marker='o', color=axcolour[4], markersize=6)
inset_ax3.plot(np.linspace(4, 18, 20), exp_marker3, color=axcolour[4])
inset_ax3.set_ylabel("$\Delta \\nu$", fontsize=23)
inset_ax3.set_xlabel("$L$", fontsize=23)
inset_ax3.set_xlim([4, 18])
inset_ax3.set_ylim([0.001, 1])
inset_ax3.set_yscale('log')
inset_ax3.tick_params(which='major', width=0.75, labelsize=23, direction='in', top=True, right=True)
inset_ax3.tick_params(which='major', length=7,  labelsize=23, direction='in', top=True, right=True)
inset_ax3.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
inset_ax3.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
# majorsy = [0.005, 0.01, 0.05, 0.1, 0.5]
# majorsy_str = ["$0.005$", "$0.01$", "$0.05$", "$0.1$", "$0.5$"]
# inset_ax3.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy_str))
# inset_ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
majorsx = [5, 7, 9, 11, 13, 15, 17]
# ax3.xaxis.set_major_formatter(ticker.FixedFormatter(majorsx_str))
inset_ax3.xaxis.set_major_locator(ticker.FixedLocator(majorsx))


right_ax3 = inset_ax3.twinx()
right_ax3.plot(inset_probX_delta3[0], inset_probY_delta3[0], marker='s', color=axcolour[5], markersize=6,label='$P_{\\rm{max}}$')
right_ax3.plot(inset_probX_delta3[1], inset_probY_delta3[1], marker='s', color=axcolour[5], markersize=6)
right_ax3.plot(inset_probX_delta3[2], inset_probY_delta3[2], marker='s', color=axcolour[5], markersize=6)
right_ax3.plot(inset_probX_delta3[3], inset_probY_delta3[3], marker='s', color=axcolour[5], markersize=6)
right_ax3.set_ylabel('$P_{\\rm{max}}$', fontsize=23)
right_ax3.set_ylim([1, 11000])
right_ax3.set_yscale('log')
right_ax3.tick_params(which='major', width=0.75, labelsize=21, direction='in', top=True, right=True)
right_ax3.tick_params(which='major', length=7,  labelsize=21, direction='in', top=True, right=True)
right_ax3.tick_params(which='minor', width=0.75, direction='in', top=True, right=True)
right_ax3.tick_params(which='minor', length=3.5, direction='in', top=True, right=True)
right_ax3.plot(np.linspace(4, 18, 20), exp_prob3, color=axcolour[5])
majorsy = [1, 10, 100, 1000, 10000]
majorsy_str = ["$10^0$", "", "$10^2$", "", "$10^4$"]
minorsy = [2,3,4,5,6,7,8,9, 20,30,40,50,60,70,80,90, 200,300,400,500,600,700,800,900,2000,3000,4000,5000,6000,7000,8000,9000]
minorsy_str = []
right_ax3.yaxis.set_major_formatter(ticker.FixedFormatter(majorsy_str))
right_ax3.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
right_ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(minorsy_str))
right_ax3.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))


# inset_ax3.legend(bbox_to_anchor=(-0.12, 1.2), loc='upper left', ncol=1, frameon=False, fontsize=20)  # bbox_to_anchor=(0.98, 0.6)
# right_ax3.legend(bbox_to_anchor=(0.45, 1.2), loc='upper left', ncol=1, frameon=False, fontsize=20)  # bbox_to_anchor=(0.98, 0.6)
#
#

plt.tight_layout()
plt.savefig("Fig3.pdf", bbox_inches="tight")
plt.show()