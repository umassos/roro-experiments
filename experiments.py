# experiment implementations for online search with switching cost
# EV charging experiments
# August 2023

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, freeze_support
import seaborn as sns
import pickle

import solar_loader as sl

import pyximport
pyximport.install()
import functions as f

import matplotlib.style as style
style.use('tableau-colorblind10')

# trace = sys.argv[1]
# slack = sys.argv[2]

# filename = ""
# if trace == "NE":
#     filename = "../carbon-traces/US-CENT-SWPP.csv"
# elif trace == "US":
#     filename = "../carbon-traces/US-NW-PACW.csv"
# elif trace == "NZ":
#     filename = "../carbon-traces/NZ-NZN.csv"
# elif trace == "CA":
#     filename = "../carbon-traces/CA-ON.csv"

# load a carbon trace
cf = pd.read_csv('CISO_direct_emissions.csv', parse_dates=True)
# cf = pd.read_csv('ERCO_direct_emissions.csv', parse_dates=True)
cf['datetime'] = pd.to_datetime(cf['UTC time'], utc=True)
cf.drop(["Unnamed: 0", "UTC time"], axis=1, inplace=True)
cf.set_index('datetime', inplace=True)

# print the types in the dataframe
# print(cf.index.dtype)
# display(cf)
# print(cf.loc['2019-08-20 09'])

############### EXPERIMENT SETUP ###############
trials = 250
# set U and L based on full data
# L, U = (cf['carbon_intensity'].min(), cf['carbon_intensity'].max())

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
# print(sum(solarData.get_generation_data().to_list()))

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

# run main experiment (just once for now)

opts = []
onemins = []
owts = []
roros = []
roroolds = []

cost_opts = []
cost_onemins = []
cost_owts = []
cost_roros = []
cost_roroolds = []

j = 0

# print(L, "   ", U)

# for each charging session, (i.e. row in the dataframe)
for i, row in enumerate(df.itertuples()):
    # randomly skip about 75% of the data
    if random.random() < 0.75:
        continue

    connect = row.connectionHour
    disconnect = row.disconnectHour

    # compute L and U based on a month of data
    monthMinus = connect - pd.Timedelta(days=20)
    L, U = (cf.loc[monthMinus:connect]['carbon_intensity'].min(), cf.loc[monthMinus:connect]['carbon_intensity'].max())
    # print(L, "   ", U)

    carbon = cf.loc[connect:disconnect]
    seq = carbon['carbon_intensity'].to_list() # get the ground truth carbon intensity for that location
    # L, U = (min(seq), max(seq))

    solar = gen.loc[connect:disconnect]
    solarSeq = solar['Solar Generation (kWh)'].to_list() # get the ground truth solar generation for that location

    # get the optimal solution
    #opt, optCost = f.optimalSolution(seq, solarSeq, beta)
    opt, optCost = f.optimalSolutionLin(seq, beta)
    if optCost > 100000000:
        continue

    # print(seq)
    # # print(solarSeq)
    # print(opt[0:len(seq)])

    # get the one min solution
    onemin, oneminCost = f.oneMinOnline(seq, solarSeq, 1, 1, 1105, beta)

    # get the one way trading solution
    owt, owtCost = f.owtOnline(seq, solarSeq, L, U, beta)

    # print(owt)

    # get the RORO solution
    roro, roroCost = f.roroOnline(seq, solarSeq, L, U, beta)

    # print(roro)

    # # get the RORO old solution
    # roroold, rorooldCost = f.roroOLDOnline(seq, L, U, beta)

    # print(roroold)
    if roroCost < optCost:
        print(seq)
        print(solarSeq)
        print(opt)
        print(owt)
        print(roro)
        optCost = roroCost
        print("OPT Cost: ", optCost)
        print("OWT Cost: ", owtCost)
        print("RORO Cost: ", roroCost)

    opts.append(opt)
    onemins.append(onemin)
    owts.append(owt)
    roros.append(roro)
    # roroolds.append(roroold)
    cost_opts.append(optCost)
    cost_onemins.append(oneminCost)
    cost_owts.append(owtCost)
    cost_roros.append(roroCost)
    # cost_roroolds.append(rorooldCost)

    j += 1
    # if j == 1000:
    #     break

###############  ##############  ###############

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_onemins = np.array(cost_onemins)
cost_owts = np.array(cost_owts)
cost_roros = np.array(cost_roros)
# cost_roroolds = np.array(cost_roroolds)

crOnemin = cost_onemins/cost_opts
crOWT = cost_owts/cost_opts
crRORO = cost_roros/cost_opts
# crROROold = cost_roroolds/cost_opts

print(crRORO)

legend = ["1-min", "OWT", "RORO"]

# CDF plot for competitive ratio (across all experiments)
plt.figure(figsize=(3.3,2.5), dpi=500)
linestyles = [":", "--", "-.", "-"]
for list in zip([crOnemin, crOWT, crRORO], linestyles):
    sns.ecdfplot(data = list[0], stat='proportion', linestyle = list[1])

plt.legend(legend)
plt.ylabel('cumulative probability')
plt.xlabel("empirical competitive ratio")
plt.tight_layout()
plt.xlim(0.9, 4)
# plt.title("CDF of Competitive Ratio, " + trace + " trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
# plt.savefig("cdf" + trace +".pdf", facecolor='w', transparent=False, bbox_inches='tight')
# plt.clf()
plt.show()

plt.figure(figsize=(3.3,2.5), dpi=500)
plt.plot(crRORO, label="RORO")
plt.plot(crOWT, label="OWT")
plt.legend()
plt.show()

# add offsets based on solar generation

# adjust "how much" work needs to be done based on the energy demand of the EV



