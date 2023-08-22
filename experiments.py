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
L, U = (cf['carbon_intensity'].min(), cf['carbon_intensity'].max())

###############  ##############  ###############

# load charging sessions
data = pd.read_json('acndata_sessions.json')
list = data['_items'].to_list()
df = pd.DataFrame(list)

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
df = df[df['duration'] > 5]

###############  ##############  ###############

# run main experiment (just once for now)

opts = []
onemins = []
owts = []
roros = []

cost_opts = []
cost_onemins = []
cost_owts = []
cost_roros = []

j = 0

print(L, "   ", U)

# for each charging session, (i.e. row in the dataframe)
for i, row in enumerate(df.itertuples()):
    connect = row.connectionHour
    disconnect = row.disconnectHour
    carbon = cf.loc[connect:disconnect]
    seq = carbon['carbon_intensity'].tolist() # get the ground truth carbon intensity for that location

    # set the switching cost
    beta = 20

    # get the optimal solution
    opt, optCost = f.optimalSolution(seq, beta)

    print(opt[0:len(seq)])

    # get the one min solution
    onemin, oneminCost = f.oneMinOnline(seq, 1, L, U, beta)

    # get the one way trading solution
    owt, owtCost = f.owtOnline(seq, L, U, beta)

    # get the RORO solution
    roro, roroCost = f.roroOnline(seq, L, U, beta)

    opts.append(opt)
    onemins.append(onemin)
    owts.append(owt)
    roros.append(roro)
    cost_opts.append(optCost)
    cost_onemins.append(oneminCost)
    cost_owts.append(owtCost)
    cost_roros.append(roroCost)

###############  ##############  ###############

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_onemins = np.array(cost_onemins)
cost_owts = np.array(cost_owts)
cost_roros = np.array(cost_roros)

crOnemin = cost_onemins/cost_opts
crOWT = cost_owts/cost_opts
crRORO = cost_roros/cost_opts

print(crOWT)

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
# plt.xlim(0, 10)
# plt.title("CDF of Competitive Ratio, " + trace + " trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
# plt.savefig("cdf" + trace +".pdf", facecolor='w', transparent=False, bbox_inches='tight')
# plt.clf()
plt.show()

# add offsets based on solar generation

# adjust "how much" work needs to be done based on the energy demand of the EV



