# experiment implementations for online conversion with switching costs
# shared functions used occasionally (no Cython)
# August 2023

import random
import math
from scipy.optimize import linprog
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import scipy.integrate as integrate
from scipy.special import lambertw
import numpy as np
import pandas as pd
import pickle

def solarOffset(carbon, solarVal, x):
    CAPACITY = 19.0

    if solarVal == 0.0: # if there is no solar, then we can't offset anything
        return carbon * x
    elif solarVal/CAPACITY > x: # if the solar is large enough to offset all of the demand
        return 1.0 * x
    else: 
        return carbon * (x - (solarVal/CAPACITY)) + 1.0 * (solarVal/CAPACITY)

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
    return -1.0 * cost

def maximizeSolution(vals, solarVals, beta, delivery, seed=None):
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
    b.append(delivery/19)

    # results = linprog(c=c, A_ub=A, b_ub=b, bounds=all_bounds, method='highs-ds')
    constraint = LinearConstraint(A, lb=b, ub=b)
    x0 = np.ones(n)
    if seed is not None:
        x0 = seed
    results = minimize(objectiveFunction, x0=x0, args=(vals, solarVals, beta), bounds=all_bounds, constraints=constraint)

    # print results
    # if results.status == 0: print(f'The solution is optimal.') 
    # print(f'Objective value: z* = {results.fun}')
    # print(f'Solution: x* = {results.x[0:n]}')
    # print(sum(results.x[0:n]))

    # return optimal variable settings + cost
    if results.status == 0:
        return results.x, results.fun
    else:
        results = minimize(objectiveFunction, x0=(np.zeros(n)), args=(vals, solarVals, beta), bounds=all_bounds, constraints=constraint)
        return results.x, results.fun