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
import seaborn as sns
import pickle
import warnings

import solar_loader as sl

import pyximport
pyximport.install()
import functions as f

import matplotlib.style as style
style.use('tableau-colorblind10')

# warnings.filterwarnings("ignore", category=ComplexWarning)
warnings.filterwarnings("ignore")

############### EXPERIMENT SETUP ###############

DC_SYSTEM_SIZE = int(sys.argv[1])

SOLAR_TILT = 30 # degrees

# set the switching cost
beta = int(sys.argv[2])

###############  ##############  ###############

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
cf = pd.read_csv('ForecastsCISO.csv', parse_dates=True)
# cf = pd.read_csv('ERCO_direct_emissions.csv', parse_dates=True)
cf['datetime'] = pd.to_datetime(cf['UTC time'], utc=True)
cf.drop(["UTC time"], axis=1, inplace=True)
cf.set_index('datetime', inplace=True)

# print the types in the dataframe
# print(cf.index.dtype)
# display(cf)
# print(cf.loc['2019-08-20 09'])

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
df = df[df['kWhDelivered'] <= 19]

###############  ##############  ###############

# run main experiment (just once for now)

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

j = 0

# print(L, "   ", U)
# 7550
# 7292
# 7499

# for each charging session, (i.e. row in the dataframe)
for i, row in enumerate(df.itertuples()):
    #randomly skip about 75% of the data
    if random.random() < 0.75:
        continue
    # if i not in [7550, 7292, 7499]:
    #     continue

    connect = row.connectionHour
    disconnect = row.disconnectHour
    delivery = row.kWhDelivered
    # delivery = 19

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
    
    # if optCost < optothCost * (delivery/19):
    #     print(seq)
    #     print(delivery/19)
    #     print(opt[:len(seq)])
    #     print(optoth[:len(seq)])
    #     print("OPT Cost: ", optCost)
    #     print("OPT other Cost: ", optothCost * (delivery/19))
    if optCost > 100000000:
        print("Error: no optimal solution found.")
        continue
    
    # get the carbon agnostic solution
    carbonAg, carbonAgCost = f.carbonAgnostic(seq, solarSeq, beta, delivery)

    # get the one min solution
    onemin, oneminCost = f.oneMinOnline(seq, solarSeq, 1, 1, 1105, beta, delivery)
    # ensure that the one min cost checks out
    assert abs(f.objectiveFunction(onemin, seq, solarSeq, beta) - oneminCost) < 0.0001, "One min cost is wrong, {} != {}".format(f.objectiveFunction(onemin, seq, solarSeq, beta), oneminCost)

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

    # print("Connection Hour: ", connect)
    # print("Disconnect Hour: ", disconnect)
    # print("kWh Delivered: ", delivery)
    # print("Sequence: ", seq)
    # print("Solar Sequence: ", solarSeq)
    # print("Optimal: ", opt[:len(seq)])
    # print("RO-Advice: ", roroAdvice)
    # print("i: {}, OPT: {}, RO-Advice: {}".format(i, optCost, roroAdviceCost))
    # print("")
    
    # if roroothCost > (roroCost * (actualdelivery/delivery)):
    #     print(seq)
    #     print(actualdelivery/delivery)
    #     print(roro)
    #     print(rorooth)
    #     print("RORO Cost: ", roroCost * (actualdelivery/delivery))
    #     print("RORO other Cost: ", roroothCost)
    # if owtothCost > (owtCost * (actualdelivery/delivery)):
    #     print(seq)
    #     print(actualdelivery/delivery)
    #     print(owt)
    #     print(owtoth)
    #     print("OWT Cost: ", owtCost * (actualdelivery/delivery))
    #     print("OWT other Cost: ", owtothCost)
    # log_roro.append(roroothCost/(roroCost * (actualdelivery/delivery)))
    # log_owt.append(owtothCost/(owtCost * (actualdelivery/delivery)))
    # log_opt.append(optothCost * (delivery/19)/optCost)

    opts.append(opt)
    onemins.append(onemin)
    owts.append(owt)
    roros.append(roro)
    roroadvices.append(roroAdvice)
    carbonags.append(carbonAg)
    # roroolds.append(roroold)
    cost_opts.append(optCost)
    cost_onemins.append(oneminCost)
    cost_owts.append(owtCost)
    cost_roros.append(roroCost)
    cost_roroadvices.append(roroAdviceCost)
    cost_carbonags.append(carbonAgCost)
    # cost_roroolds.append(rorooldCost)

    j += 1
    # if j == 1000:
    #     break

###############  ##############  ###############

# print(sum(log_roro)/len(log_roro))
# print(sum(log_owt)/len(log_owt))
# print(sum(log_opt)/len(log_opt))

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_onemins = np.array(cost_onemins)
cost_owts = np.array(cost_owts)
cost_roros = np.array(cost_roros)
cost_roroadvices = np.array(cost_roroadvices)
cost_carbonags = np.array(cost_carbonags)
# cost_roroolds = np.array(cost_roroolds)

crOnemin = cost_onemins/cost_opts
crOWT = cost_owts/cost_opts
crRORO = cost_roros/cost_opts
crROROAdvice = cost_roroadvices/cost_opts
crCarbonAg = cost_carbonags/cost_opts
# crROROold = cost_roroolds/cost_opts

print(crCarbonAg)

legend = ["Carbon Agnostic", "Simple Threshold", "OWT", "RORO", "RO-Advice ($\lambda = 0.5$)"]

# CDF plot for competitive ratio (across all experiments)
plt.figure(figsize=(3.8,3), dpi=300)
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
plt.xlim(1, 4.5)
if DC_SYSTEM_SIZE > 0:
    plt.xlim(1, 6)
# plt.title("CDF of Competitive Ratio, " + trace + " trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
plt.savefig("cdf_s{}_b{}.png".format(DC_SYSTEM_SIZE, beta), facecolor='w', transparent=False, bbox_inches='tight')
plt.clf()
# plt.show()

# plt.figure(figsize=(3.3,2.5), dpi=300)
# plt.plot(crRORO, label="RORO")
# plt.plot(crOWT, label="OWT")
# plt.plot(crROROAdvice, label="RORO Advice")
# plt.legend()
# plt.show()
# plt.figure(figsize=(3.3,2.5), dpi=300)
# plt.plot(crRORO, label="RORO")
# plt.plot(crOWT, label="OWT")
# plt.plot(crROROAdvice, label="RORO Advice")
# plt.legend()
# plt.show()

# add offsets based on solar generation

# adjust "how much" work needs to be done based on the energy demand of the EV



