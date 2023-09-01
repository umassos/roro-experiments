# experiment implementations for online search with switching cost
# EV charging experiments -- varying advice error
# September 2023

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
# style.use('tableau-colorblind10')

# load a carbon trace
cf = pd.read_csv('ForecastsCISO.csv', parse_dates=True)
# cf = pd.read_csv('ERCO_direct_emissions.csv', parse_dates=True)
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
df = df[df['kWhDelivered'] <= 19]

###############  ##############  ###############

# run main experiment (just once for now)

opts = []
roros = []
advices = []
roroadvices = []
roroadvices2 = []
roroadvices3 = []

cost_opts = []
cost_roros = []
cost_advices = []
cost_roroadvices = []
cost_roroadvices2 = []
cost_roroadvices3 = []

j = 0

# for each charging session, (i.e. row in the dataframe)
for i, row in enumerate(df.itertuples()):
    # randomly skip about 75% of the data
    if random.random() < 0.75:
        continue

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

    # generate a "predicted sequence" (real sequence plus noise)
    predseq = f.addNoise(seq, noiseFactor=0.5)

    solar = gen.loc[connect:disconnect]
    solarSeq = solar['Solar Generation (kWh)'].to_list() # get the ground truth solar generation for that location

    if len(solarSeq) < len(seq):
        continue 
    
    # get the optimal solution and the predicted optimal
    if DC_SYSTEM_SIZE == 0:
        opt, optCost = f.optimalSolutionLin(seq, beta, delivery)
        predOpt, predOptCost = f.optimalSolutionLin(predseq, beta, delivery)
    else:
        opt, optCost = f.optimalSolution(seq, solarSeq, beta, delivery)
        predOpt, predOptCost = f.optimalSolution(predseq, solarSeq, beta, delivery)
    
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

    # get the RORO solution
    roro, roroCost = f.roroOnline(seq, solarSeq, L, U, beta, delivery)

    # if (roroCost < optCost) and DC_SYSTEM_SIZE > 0:
    #     # try again
    #     costs = [oneminCost, owtCost, roroCost]
    #     bestSoFar = np.argmin(costs)
    #     bestSolutionSoFar = [onemin, owt, roro][bestSoFar]
    #     assert abs(f.objectiveFunction(bestSolutionSoFar, seq, solarSeq, beta) - costs[bestSoFar]) < 0.0001, "Best cost is wrong, {} != {}".format(f.objectiveFunction(bestSolutionSoFar, seq, solarSeq, beta), costs[bestSoFar])
        
    #     print("retrying optimal, current cost = {}, and best so far ({}) = {}".format(optCost, bestSoFar, costs[bestSoFar]))
    #     opt, optCost = f.optimalSolution(seq, solarSeq, beta, delivery, seed=bestSolutionSoFar)
    #     print("new cost = {}".format(optCost))

    # get the learning-augmented RORO solution
    roroAdvice, roroAdviceCost = f.convexComb(seq, solarSeq, beta, 0.75, predOpt.tolist(), roro)

    # get the learning-augmented RORO solution
    roroAdvice2, roroAdviceCost2 = f.convexComb(seq, solarSeq, beta, 0.5, predOpt.tolist(), roro)

    # get the learning-augmented RORO solution
    roroAdvice3, roroAdviceCost3 = f.convexComb(seq, solarSeq, beta, 0.25, predOpt.tolist(), roro)
    
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
    roros.append(roro)
    advices.append(predOpt)
    roroadvices.append(roroAdvice)
    roroadvices2.append(roroAdvice2)
    roroadvices3.append(roroAdvice3)
    cost_opts.append(optCost)
    cost_roros.append(roroCost)
    cost_advices.append(predOptCost)
    cost_roroadvices.append(roroAdviceCost)
    cost_roroadvices2.append(roroAdviceCost2)
    cost_roroadvices3.append(roroAdviceCost3)

    j += 1
    # if j == 1000:
    #     break

###############  ##############  ###############

# print(sum(log_roro)/len(log_roro))
# print(sum(log_owt)/len(log_owt))
# print(sum(log_opt)/len(log_opt))

# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_roros = np.array(cost_roros)
cost_advices = np.array(cost_advices)
cost_roroadvices = np.array(cost_roroadvices)
cost_roroadvices2 = np.array(cost_roroadvices2)
cost_roroadvices3 = np.array(cost_roroadvices3)

crRORO = cost_roros/cost_opts
crAdvice = cost_advices/cost_opts
crROROAdvice = cost_roroadvices/cost_opts
crROROAdvice2 = cost_roroadvices2/cost_opts
crROROAdvice3 = cost_roroadvices3/cost_opts

print(crRORO)

legend = ["RORO", "BB Advice", "ROAdvice ($\lambda = 0.75$)", "ROAdvice ($\lambda = 0.5$)", "ROAdvice ($\lambda = 0.25$)"]

# CDF plot for competitive ratio (across all experiments)
plt.figure(figsize=(3.8,3), dpi=300)
linestyles = [":", "--", "-.", "-", ":"]
for list in zip([crRORO, crAdvice, crROROAdvice, crROROAdvice2, crROROAdvice3], linestyles):
    sns.ecdfplot(data = list[0], stat='proportion', linestyle = list[1])

plt.legend(legend)
plt.ylabel('cumulative probability')
plt.xlabel("empirical competitive ratio")
plt.tight_layout()
plt.xlim(0.9, 8)
# plt.title("CDF of Competitive Ratio, " + trace + " trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
# plt.savefig("cdf.png", facecolor='w', transparent=False, bbox_inches='tight')
# plt.clf()
plt.show()

# plt.figure(figsize=(3.3,2.5), dpi=300)
# plt.plot(crRORO, label="RORO")
# plt.plot(crOWT, label="OWT")
# plt.plot(crROROAdvice, label="RORO Advice")
# plt.legend()
# plt.show()

# add offsets based on solar generation

# adjust "how much" work needs to be done based on the energy demand of the EV



