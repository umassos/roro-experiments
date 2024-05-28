# experiment implementations for online conversion with switching costs
# EV charging experiments -- varying advice error
# September 2023

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, Manager, freeze_support
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.style as style

style.use('seaborn-v0_8') #sets the size of the charts

import supplemental as s
import solar_loader as sl

import pyximport
pyximport.install()
import functions2 as f

# load a carbon trace
cf = pd.read_csv('ForecastsCISO.csv', parse_dates=True)
cf['datetime'] = pd.to_datetime(cf['UTC time'], utc=True)
cf.drop(["UTC time"], axis=1, inplace=True)
cf.set_index('datetime', inplace=True)

############### EXPERIMENT SETUP ###############

DC_SYSTEM_SIZE = 0 # kW
SOLAR_TILT = 30 # degrees

# set the switching cost
beta = 20

###############  ##############  ###############

# load charging sessions
data = pd.read_json('acndata_sessions.json')
list = data['_items'].to_list()
df = pd.DataFrame(list)

# load solar data

solarData = sl.SolarData(SOLAR_TILT, DC_SYSTEM_SIZE)
gen = solarData.get_full_data()
# set index to utc
gen.index = gen.index.tz_localize("UTC")
# add 7 hours
gen.index = gen.index + pd.DateOffset(hours=7)

# clean data
df["connectionTime"] = pd.to_datetime(df["connectionTime"])
df["disconnectTime"] = pd.to_datetime(df["disconnectTime"])

# round each time to the last hour
df["connectionHour"] = df["connectionTime"].dt.floor("H")
# round each time to the next hour
df["disconnectHour"] = df["disconnectTime"].dt.ceil("H")
# compute duration of charging sessions
df["duration"] = (df["disconnectHour"] - df["connectionHour"]).dt.seconds / 3600

# drop unnecessary columns
df.drop(["connectionTime", "disconnectTime", "sessionID", "clusterID", "siteID", "userID", "userInputs"], axis=1, inplace=True)

# only sufficiently long charging sessions
df = df[df['duration'] >= 5]

###############  ##############  ###############

def adv_experiment(df, beta, factor):
    opts = []
    roros = []
    advices = []
    roroadvices = []
    roroadvices2 = []
    roroadvices3 = []
    roroadvices4 = []

    cost_opts = []
    cost_roros = []
    cost_advices = []
    cost_roroadvices = []
    cost_roroadvices2 = []
    cost_roroadvices3 = []
    cost_roroadvices4 = []

    # for each charging session, (i.e. row in the dataframe)
    for i, row in enumerate(df.itertuples()):
        # randomly skip about 75% of the data (comment this out for final plots)
        if random.random() < 0.75:
            continue

        connect = row.connectionHour.tz_localize("UTC")
        disconnect = row.disconnectHour.tz_localize("UTC")
        delivery = row.kWhDelivered

        # compute L and U based on a month of data
        monthMinus = (connect - pd.Timedelta(days=20))
        if monthMinus < cf.index[0]:
            monthMinus = cf.index[0]
        L, U = (cf.loc[monthMinus:connect]['carbon_intensity'].min(), cf.loc[monthMinus:connect]['carbon_intensity'].max())
        # print(L, "   ", U)
        
        carbon = cf.loc[connect:disconnect]
        seq = carbon['carbon_intensity'].to_list() # get the ground truth carbon intensity for that location

        solar = gen.loc[connect:disconnect]
        solarSeq = solar['Solar Generation (kWh)'].to_list() # get the ground truth solar generation for that location

        if len(solarSeq) < len(seq):
            continue 
        
        # get the optimal solution and the predicted optimal
        if DC_SYSTEM_SIZE == 0:
            opt, optCost = f.optimalSolutionLin(seq, beta, delivery)
            # adversary is the solution that maximizes the objective function
            adversary, _ = s.maximizeSolution(seq, solarSeq, beta, delivery)
        else:
            opt, optCost = f.optimalSolution(seq, solarSeq, beta, delivery)
            adversary, _ = s.maximizeSolution(seq, solarSeq, beta, delivery)
        
        # pred Opt is a convex combination of the optimal and the adversary
        predOpt = f.combine(opt[:len(adversary)], adversary, factor)
        predOptCost = f.objectiveFunction(predOpt, seq, solarSeq, beta)

        # ad hoc way of checking for convergence (this shouldn't ever happen)
        if optCost > 100000000:
            print("Error: no optimal solution found.")
            continue

        # get the RORO solution
        roro, roroCost = f.roroOnline(seq, solarSeq, L, U, beta, delivery)

        # get the learning-augmented RORO solution for different lambdas
        roroAdvice, roroAdviceCost = f.convexComb(seq, solarSeq, beta, 0.8, predOpt, roro)

        roroAdvice2, roroAdviceCost2 = f.convexComb(seq, solarSeq, beta, 0.6, predOpt, roro)

        roroAdvice3, roroAdviceCost3 = f.convexComb(seq, solarSeq, beta, 0.4, predOpt, roro)

        roroAdvice4, roroAdviceCost4 = f.convexComb(seq, solarSeq, beta, 0.2, predOpt, roro)
        
        opts.append(opt)
        roros.append(roro)
        advices.append(predOpt)
        roroadvices.append(roroAdvice)
        roroadvices2.append(roroAdvice2)
        roroadvices3.append(roroAdvice3)
        roroadvices4.append(roroAdvice4)
        cost_opts.append(optCost)
        cost_roros.append(roroCost)
        cost_advices.append(predOptCost)
        cost_roroadvices.append(roroAdviceCost)
        cost_roroadvices2.append(roroAdviceCost2)
        cost_roroadvices3.append(roroAdviceCost3)
        cost_roroadvices4.append(roroAdviceCost4)


    ###############  ##############  ###############

    # compute competitive ratios
    cost_opts = np.array(cost_opts)
    cost_roros = np.array(cost_roros)
    cost_advices = np.array(cost_advices)
    cost_roroadvices = np.array(cost_roroadvices)
    cost_roroadvices2 = np.array(cost_roroadvices2)
    cost_roroadvices3 = np.array(cost_roroadvices3)
    cost_roroadvices4 = np.array(cost_roroadvices4)

    crRORO = cost_roros/cost_opts
    crAdvice = cost_advices/cost_opts
    crROROAdvice = cost_roroadvices/cost_opts
    crROROAdvice2 = cost_roroadvices2/cost_opts
    crROROAdvice3 = cost_roroadvices3/cost_opts
    crROROAdvice4 = cost_roroadvices4/cost_opts

    return crRORO, crAdvice, crROROAdvice, crROROAdvice2, crROROAdvice3, crROROAdvice4

# plot heatmap down here
algorithms = ["RORO", "ROAdvice [$\\epsilon \\sim 2.97$]", "ROAdvice [$\\epsilon \\sim 2.23$]", "ROAdvice [$\\epsilon \\sim 1.48$]", "ROAdvice [$\\epsilon \\sim 0.74$]", "black-box advice"]
adversaryFactors = np.linspace(0, 1, 21)

print(adversaryFactors)
time.sleep(5)
cr = np.array([[0.0 for _ in range(0, len(adversaryFactors))] for _ in range(0, 6)])
print(cr)

roro = []
roroAdvice = []
roroAdvice2 = []
roroAdvice3 = []
roroAdvice4 = []
advice = []

for i, factor in enumerate(adversaryFactors):
    crRORO, crAdvice, crROROAdvice, crROROAdvice2, crROROAdvice3, crROROAdvice4 = adv_experiment(df, beta, factor)
    cr[0][i] = crRORO.mean()
    cr[1][i] = crROROAdvice4.mean()
    cr[2][i] = crROROAdvice3.mean()
    cr[3][i] = crROROAdvice2.mean()
    cr[4][i] = crROROAdvice.mean()
    cr[5][i] = crAdvice.mean()
    roro.append(crRORO)
    roroAdvice.append(crROROAdvice)
    roroAdvice2.append(crROROAdvice2)
    roroAdvice3.append(crROROAdvice3)
    roroAdvice4.append(crROROAdvice4)
    advice.append(crAdvice)



fig, ax = plt.subplots(figsize=(5,2), dpi=300)
# make the matrix expand to fill the available area
im = ax.imshow(cr, cmap='rainbow', interpolation='nearest', aspect='auto', alpha=0.75)
# white grid lines
ax.set_xticks(np.arange(-.5, 21, 1), minor=True)
ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.set_xlabel("adversarial factor ($\\zeta$)")
ax.grid(which='major', color='w', linestyle='-', linewidth=0)
ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
ax.tick_params(which='minor', bottom=False, left=False)

# Show all ticks and label them with the respective list entries
# labels should be printed to two decimal places
ax.set_xticks(np.arange(len(adversaryFactors)), labels=adversaryFactors)
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
ax.set_yticks(np.arange(len(algorithms)), labels=algorithms)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(),
         rotation_mode="anchor")

# show a colorbar on the side (vertical orientation)
# make the colorbar the same height as the image
cbar = ax.figure.colorbar(im, ax=ax, orientation='vertical', label="Competitive Ratio")

# ax.set_title("Competitive Ratio of different algorithms")
fig.tight_layout()
plt.savefig("advice_plots/grid_advice.png", facecolor='w', transparent=False, bbox_inches='tight')

# linestyles = ["-.", ":", "--", (0, (3, 1, 1, 1, 1, 1)), "-", "-."]

# # do each of the 4 lines separately in its own subplot
# fig, axs = plt.subplots(2, 2, figsize=(5,2), dpi=300)

# lines = []
# # for i, array in enumerate([roro, roroAdvice4, roroAdvice3, roroAdvice2, roroAdvice, advice]):
# # for i, array in enumerate([roro, roroAdvice3, roroAdvice2, advice]):
# algorithms = ["ROAdvice [$\\epsilon \\sim 2.97$]", "ROAdvice [$\\epsilon \\sim 2.23$]", "ROAdvice [$\\epsilon \\sim 1.48$]", "black-box advice"]

# axs[0,0].plot(adversaryFactors, np.mean(roroAdvice4, axis=1), linestyle=linestyles[0])
# axs[0,0].plot(adversaryFactors, np.mean(roro, axis=1), linestyle=linestyles[4])
# axs[0,0].fill_between(adversaryFactors,  np.mean(roroAdvice4, axis=1)-np.std(roroAdvice4, axis=1), np.mean(roroAdvice4, axis=1)+np.std(roroAdvice4, axis=1), alpha=0.3)

# axs[0,1].plot(adversaryFactors, np.mean(roroAdvice3, axis=1), linestyle=linestyles[1])
# axs[0,1].plot(adversaryFactors, np.mean(roro, axis=1), linestyle=linestyles[4])
# axs[0,1].fill_between(adversaryFactors,  np.mean(roroAdvice3, axis=1)-np.std(roroAdvice3, axis=1), np.mean(roroAdvice3, axis=1)+np.std(roroAdvice3, axis=1), alpha=0.3)

# axs[1,0].plot(adversaryFactors, np.mean(roroAdvice2, axis=1), linestyle=linestyles[2])
# axs[1,0].plot(adversaryFactors, np.mean(roro, axis=1), linestyle=linestyles[4])
# axs[1,0].fill_between(adversaryFactors,  np.mean(roroAdvice2, axis=1)-np.std(roroAdvice2, axis=1), np.mean(roroAdvice2, axis=1)+np.std(roroAdvice2, axis=1), alpha=0.3)

# axs[1,1].plot(adversaryFactors, np.mean(advice, axis=1), linestyle=linestyles[3])
# axs[1,1].plot(adversaryFactors, np.mean(roro, axis=1), linestyle=linestyles[4])
# axs[1,1].fill_between(adversaryFactors,  np.mean(advice, axis=1)-np.std(advice, axis=1), np.mean(advice, axis=1)+np.std(advice, axis=1), alpha=0.3)


# # four column legend at bottom of plot
# # plt.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
# for x,y in [(0,0), (0,1), (1,0), (1,1)]:
#     axs[x,y].set_ylim(1, 2.5)
#     axs[x,y].set_xlabel("$\\zeta$")
#     axs[x,y].set_ylabel("comp. ratio")

# axs[0,0].set_title(algorithms[0])
# axs[0,1].set_title(algorithms[1])
# axs[1,0].set_title(algorithms[2])
# axs[1,1].set_title(algorithms[3])
# # fig.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.3))
# fig.tight_layout()
# plt.show()



