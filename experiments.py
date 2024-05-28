# experiment implementations for online conversion with switching costs
# EV charging experiments
# August 2023

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings

import solar_loader as sl

import pyximport
pyximport.install()
import functions2 as f

import matplotlib.style as style
style.use('tableau-colorblind10')
warnings.filterwarnings("ignore")

############### EXPERIMENT SETUP ###############

DC_SYSTEM_SIZE = int(sys.argv[1])

SOLAR_TILT = 30 # degrees

# set the switching cost
beta = int(sys.argv[2])

###############  ##############  ###############

# load the carbon trace
cf = pd.read_csv('ForecastsCISO.csv', parse_dates=True)
# cf = pd.read_csv('ERCO_direct_emissions.csv', parse_dates=True)
cf['datetime'] = pd.to_datetime(cf['UTC time'], utc=True)
cf.drop(["UTC time"], axis=1, inplace=True)
cf.set_index('datetime', inplace=True)

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
df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True)
df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True)

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

# run main experiment 

opts = []
onemins = []
owts = []
roros = []
roroadvices = []
carbonags = []

cost_opts = []
cost_onemins = []
cost_owts = []
cost_roros = []
cost_roroadvices = []
cost_carbonags = []

# for each charging session, (i.e. row in the dataframe)
for i, row in tqdm(enumerate(df.itertuples())):
    #randomly skip most of the data (comment this out for final plots)
    if random.random() < 0.9:
        continue

    connect = row.connectionHour
    disconnect = row.disconnectHour
    delivery = row.kWhDelivered

    # compute L and U based on a month of data
    monthMinus = connect - pd.Timedelta(days=20)
    if monthMinus < cf.index[0]:
        monthMinus = cf.index[0]
    L, U = (cf.loc[monthMinus:connect]['carbon_intensity'].min(), cf.loc[monthMinus:connect]['carbon_intensity'].max())
    # print(L, "   ", U)
    
    carbon = cf.loc[connect:disconnect]
    seq = carbon['carbon_intensity'].to_list() # get the ground truth carbon intensity for that location
    predseq = carbon['avg_carbon_intensity_forecast'].to_list()
    #  L, U = (min(seq), max(seq))

    solar = gen.loc[connect:disconnect]
    solarSeq = solar['Solar Generation (kWh)'].to_list() # get the ground truth solar generation for that location

    if len(solarSeq) < len(seq):
        continue 
    
    # get the optimal solution and the predicted optimal
    if DC_SYSTEM_SIZE == 0:
        opt, optCost = f.optimalSolutionLin(seq, beta, delivery)
        predOpt, _ = f.optimalSolutionLin(predseq, beta, delivery)
        predOptCost = f.objectiveFunction(predOpt, seq, solarSeq, beta)
    else:
        opt, optCost = f.optimalSolution(seq, solarSeq, beta, delivery)
        predOpt, _ = f.optimalSolution(predseq, solarSeq, beta, delivery)
        predOptCost = f.objectiveFunction(predOpt, seq, solarSeq, beta)
    
    if optCost > 100000000:
        print("Error: no optimal solution found.")
        continue
    
    # get the carbon agnostic solution
    carbonAg, carbonAgCost = f.carbonAgnostic(seq, solarSeq, beta, delivery)

    # get the one min solution
    onemin, oneminCost = f.oneMinOnline(seq, solarSeq, 1, L, U, beta, delivery)

    # get the one way trading solution
    owt, owtCost = f.owtOnline(seq, solarSeq, L, U, beta, delivery)

    # get the RORO solution
    roro, roroCost = f.roroOnline(seq, solarSeq, L, U, beta, delivery)

    if (oneminCost < optCost or owtCost < optCost or roroCost < optCost) and DC_SYSTEM_SIZE > 0:
        # try again
        costs = [oneminCost, owtCost, roroCost]
        bestSoFar = np.argmin(costs)
        bestSolutionSoFar = [onemin, owt, roro][bestSoFar]
        assert abs(f.objectiveFunction(bestSolutionSoFar, seq, solarSeq, beta) - costs[bestSoFar]) < 0.0001, "Best cost is wrong, {} != {}".format(f.objectiveFunction(bestSolutionSoFar, seq, solarSeq, beta), costs[bestSoFar])
        
        # print("retrying optimal, current cost = {}, and best so far ({}) = {}".format(optCost, bestSoFar, costs[bestSoFar]))
        opt, optCost = f.optimalSolution(seq, solarSeq, beta, delivery, seed=bestSolutionSoFar)
        # print("new cost = {}".format(optCost))

    # get the learning-augmented RORO solution
    roroAdvice, roroAdviceCost = f.convexComb(seq, solarSeq, beta, 0.5, predOpt.tolist(), roro)

    opts.append(opt)
    onemins.append(onemin)
    owts.append(owt)
    roros.append(roro)
    roroadvices.append(roroAdvice)
    carbonags.append(carbonAg)
    cost_opts.append(optCost)
    cost_onemins.append(oneminCost)
    cost_owts.append(owtCost)
    cost_roros.append(roroCost)
    cost_roroadvices.append(roroAdviceCost)
    cost_carbonags.append(carbonAgCost)


###############  ##############  ###############

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_onemins = np.array(cost_onemins)
cost_owts = np.array(cost_owts)
cost_roros = np.array(cost_roros)
cost_roroadvices = np.array(cost_roroadvices)
cost_carbonags = np.array(cost_carbonags)

crOnemin = cost_onemins/cost_opts
crOWT = cost_owts/cost_opts
crRORO = cost_roros/cost_opts
crROROAdvice = cost_roroadvices/cost_opts
crCarbonAg = cost_carbonags/cost_opts

print(crCarbonAg)

legend = ["Carbon Agnostic", "Simple Threshold", "OWT", "RORO", "RO-Advice ($\lambda = 0.5$)"]

# CDF plot for competitive ratio (across all experiments)
plt.figure(figsize=(3.4834,2.75), dpi=300)
linestyles = ["-.", ":", "--", (0, (3, 1, 1, 1, 1, 1)), "-"]
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

for list in zip([crCarbonAg, crOnemin, crOWT, crRORO, crROROAdvice], linestyles):
    sns.ecdfplot(data = list[0], stat='proportion', linestyle = list[1])

plt.legend(legend)
plt.ylabel('cumulative probability')
plt.xlabel("empirical competitive ratio")
plt.tight_layout()
plt.xlim(1, 3.5)
if DC_SYSTEM_SIZE > 0:
    plt.xlim(1, 6)
plt.savefig("plots/cdf_s{}_b{}.png".format(DC_SYSTEM_SIZE, beta), facecolor='w', transparent=False, bbox_inches='tight')
plt.clf()

# print mean and 95th percentile of each competitive ratio
print("Solar: {}, Switching Cost: {}".format(DC_SYSTEM_SIZE, beta))
print("Carbon Agnostic: ", np.mean(crCarbonAg), np.percentile(crCarbonAg, 95))
print("Simple Threshold: ", np.mean(crOnemin), np.percentile(crOnemin, 95))
print("OWT: ", np.mean(crOWT), np.percentile(crOWT, 95))
print("RORO: ", np.mean(crRORO), np.percentile(crRORO, 95))
print("RO-Advice: ", np.mean(crROROAdvice), np.percentile(crROROAdvice, 95))



