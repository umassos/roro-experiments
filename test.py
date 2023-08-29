import time

from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.integrate as integrate
from scipy.special import lambertw

import numpy as np
import math

import pyximport
import sys
pyximport.install()

import functions as f

def solarOffset(carbon, solarVal, x):
    CAPACITY = 19.0

    if solarVal == 0.0: # if there is no solar, then we can't offset anything
        return carbon * x
    elif solarVal/CAPACITY > x: # if the solar is large enough to offset all of the demand
        return 0.0
    else: 
        return carbon * (x - (solarVal/CAPACITY))    

def objectiveFunction(vars, vals, solarVals, beta):
    cost = 0.0
    n = len(vals)
    for (i, carbon) in enumerate(vals):
        solar = solarVals[i]
        cost += solarOffset(carbon, solar, vars[i])

        # add switching cost
        if i == 0:
            cost += beta * abs(vars[i] - 0.0)
        elif i == n-1:
            cost += beta * abs(vars[i] - vars[i-1])
            cost += beta * abs(0.0 - vars[i])
        else:
            cost += beta * abs(vars[i] - vars[i-1])
    return cost

def optimalSolution(vals, solarVals, beta):
    n = len(vals)
    all_bounds = [(0,1) for _ in range(0, n)]

    # declare inequality constraint matrix (2n + 1) x (2n + 1)
    # and the inequality constraint vector (2n + 1)
    A = []
    b = []
    # append first row (deadline constraint)
    row = [0 for i in range(0, n)]
    for i in range(0, n):
        row[i] = 1
    A.append(row)
    b.append(1)

    # results = linprog(c=c, A_ub=A, b_ub=b, bounds=all_bounds, method='highs-ds')
    constraint = LinearConstraint(A, lb=b, ub=b)
    results = minimize(objectiveFunction, x0=(np.ones(n)), args=(vals, solarVals, beta), bounds=all_bounds, constraints=constraint)

    # return optimal variable settings + cost
    if results.status == 0:
        # print(sum(results.x))
        # print(results.x)
        return results.x, results.fun
    else:
        print("Error: no optimal solution found.")
        return [], math.inf

if __name__ == '__main__':
    # generate a synthetic sequence
    # seq = f.generateSyntheticSequence(10, 30, 1)
    # print(len(seq))
    # print(seq)
    # solar_seq = [1.9 for _ in seq]
    # for i in range(0, int(len(solar_seq)/2)):
    #     solar_seq[i] = 0
    # solar_seq[10] = 0.9
    # print(solar_seq)

    # I'm going to manually define a toy sequence here
    seq         = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    solar       = [0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0.5]
    solar_seq   = [x * 19 for x in solar]

    # set the switching cost
    beta = 2

    print(objectiveFunction(solar, seq, solar_seq, beta))
    print(sum(solar))

    print("Python Native")
    # get the optimal solution
    opt, optCost = optimalSolution(seq, solar_seq, beta)
    print(optCost)
    print(opt)

    print("Cython")
    # get the optimal solution
    opt, optCost = f.optimalSolution(seq, solar_seq, beta)
    print(optCost)
    print(opt)

    # print(optCost)
    # for i in range(0, len(opt)):
    #     # print all on one line
    #     sys.stdout.write(f'{opt[i]:.6f}' + " ")

    # # get the one min solution
    # onemin, oneminCost = f.oneMinOnline(seq, 1, min(seq), max(seq), beta)

    # # get the one way trading solution
    # owt, owtCost = f.owtOnline(seq, min(seq), max(seq), beta)

    # # get the RORO solution
    # roro, roroCost = f.roroOnline(seq, min(seq), max(seq), beta)

    # # print the sequence
    # print("Sequence: ", seq)

    # # print the optimal solution and cost
    # print("OPT: ", opt[0:len(seq)], " OPT Cost: ", optCost)
    # # print("OPT switching: ", opt[len(seq):])

    # # print the one min solution and cost
    # print("1-min: ", onemin, " 1-min Cost: ", oneminCost)

    # # print the one way trading solution and cost
    # print("OWT: ", owt, " OWT Cost: ", owtCost)

    # # print the RORO solution and cost
    # print("RORO: ", roro, " RORO Cost: ", roroCost)


