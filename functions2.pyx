# experiment implementations for online conversion with switching costs
# shared function implementations in Cython
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

def addNoise(vals, noiseFactor):
    vals = np.array(vals)
    mean = np.mean(vals)
    delta = vals - mean
    # noise = np.random.normal(loc=0.0, scale=noiseFactor, size=len(vals))
    noisyInterval = np.full_like(vals, mean) + (noiseFactor*delta)
    noisyInterval = noisyInterval.clip(min=0)
    return noisyInterval.tolist()

def combine(vals, vals2, mixing):
    values = np.array(vals)
    values2 = np.array(vals2)
    newSeq = (1-mixing)*values + mixing*values2
    return newSeq.tolist()

cdef float cosine(float x):
    return (5.1 + 5 * math.cos(x/3.8))

cpdef list generateSyntheticSequence(float start, float end, float interval):
    cdef list sequence
    cdef float cur, noise
    
    sequence = []

    cur = start
    while (cur < end):
        sequence.append(cosine(cur))
        cur += interval

    # noise = np.random.normal(0,noiseFactor,len(sequence))
    
    return sequence

def solarOffset(carbon, solarVal, x):
    CAPACITY = 19.0

    if solarVal == 0.0: # if there is no solar, then we can't offset anything
        return carbon * x
    elif solarVal/CAPACITY > x: # if the solar is large enough to offset all of the demand
        return 1.0 * x
    else: 
        return carbon * (x - (solarVal/CAPACITY)) + 1.0 * (solarVal/CAPACITY)

# objectiveFunction computes the minimization objective function for the OCS problem.
# vars is the time series of decision variables (length T)
# vals is the time series of carbon intensity values (length T)
# beta is a fixed switching cost (you can set this to 0)
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

cpdef tuple[list, float] optimalSolutionLin(list vals, float beta, float delivery):
    cdef int n, dim, i
    cdef list c, A, b, row, all_bounds

    n = len(vals)
    dim = 2*n + 1
    all_bounds = [(0,1) for i in range(0, dim)]

    c = []
    # declare coefficients of the objective function (2n + 1)
    for i in range(0, n):
        c.append(vals[i])
    for i in range(0, n+1):
        c.append(beta)

    # declare inequality constraint matrix (2n + 1) x (2n + 1)
    # and the inequality constraint vector (2n + 1)
    A = []
    b = []
    # append first row (deadline constraint)
    row = [0 for i in range(0, dim)]
    for i in range(0, n):
        row[i] = -1
    A.append(row)
    b.append(-(delivery/19))

    # append subsequent rows (switching cost constraint)
    row = [0 for i in range(0, dim)]
    row[0] = 1
    row[n] = -1
    A.append(row)
    b.append(0)
    for i in range(0, n-1):
        row = [0 for i in range(0, dim)]
        row[i] = -1
        row[i+1] = 1
        row[i+n+1] = -1
        A.append(row.copy())
        b.append(0)

        row[i] = 1
        row[i+1] = -1
        row[i+n+1] = -1
        A.append(row.copy())
        b.append(0)
    row = [0 for i in range(0, dim)]
    row[n-1] = 1
    row[-1] = -1
    A.append(row)
    b.append(0)

    results = linprog(c=c, A_ub=A, b_ub=b, bounds=all_bounds, method='highs-ds')

    # print results
    # if results.status == 0: print(f'The solution is optimal.') 
    # print(f'Objective value: z* = {results.fun}')
    # print(f'Solution: x* = {results.x}')

    # return optimal variable settings + cost
    if results.status == 0:
        return results.x, results.fun
    else:
        print("Error: no optimal solution found.")
        return [], math.inf

def optimalSolution(vals, solarVals, beta, delivery, seed=None):
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

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] oneMinOnline(list vals, list solarVals, int k, float L, float U, float beta, float delivery):
    cdef int lastElem, i
    cdef bint accept
    cdef list sol
    cdef float cost, threshold, val, accepted
    
    prevAccepted = False
    sol = []
    accepted = 0
    lastElem = len(vals)

    threshold = math.sqrt(1*1105)

    total = delivery/19

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= total:
            sol.append(0)
            continue
        
        if (len(vals)-i) == int(np.ceil(total-accepted)): # must accept all remaining elements
            lastElem = i
            break

        solar = solarVals[i]
        val = solarOffset(carbon, solar, ((1/k) * min(total-accepted, 1)))
        
        accept = (val <= threshold * 1/k)
        if accept:
            sol.append((1/k) * min(total-accepted, 1))
            accepted += min(total-accepted, 1)
        else:
            sol.append(0)

    if accepted < total:
        for i in range(lastElem, len(vals)):
            solar = solarVals[i]
            carbon = vals[i]
            sol.append((1/k) * min(total-accepted, 1))
            accepted += min(total-accepted, 1)

    assert sum(sol) == total, "solution does not add up to total"

    cost = objectiveFunction(sol, vals, solarVals, beta)
    return sol, cost


# standard one-way trading algorithm implementation

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] owtOnline(list vals, list solarVals, float L, float U, float beta, float delivery):
    cdef int i
    cdef list sol
    cdef float carbon, alpha, accepted, val, solar

    sol = []
    accepted = 0.0
    total = delivery/19

    # get value for alpha
    alpha = 1 / (1 + lambertw( ( (L/U) - 1 ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= 1:
            sol.append(0)
            continue
        
        solar = solarVals[i]

        remainder = (1 - accepted) * total
        
        if i == len(vals) - int(np.ceil(remainder)): # must accept last price(s)
            amount = min(remainder, 1)
            val = solarOffset(carbon, solar, amount)
            sol.append(amount)
            accepted += amount/total
            j = 1
            remainder = (1 - accepted) * total
            while remainder > 0:
                amount = min(remainder, 1)
                carbon = vals[i+j]
                solar = solarVals[i+j]
                val = solarOffset(carbon, solar, amount)
                sol.append(amount)
                accepted += amount/total
                j += 1
                remainder = (1 - accepted) * total
            break
        
        # solve for threshold-defined amount
        previous = 0
        if i != 0:
            previous = sol[-1]
        amount = owtHelper(carbon, solar, accepted, alpha, L, U, previous)
        amount = min(total * amount, 1)
        accepted += amount/total
        val = solarOffset(carbon, solar, amount)
        sol.append(amount)

    cost = objectiveFunction(sol, vals, solarVals, beta)
    return sol, cost

# helper for one-way trading algorithm
cpdef float owtHelper(float carbon, float solar, float accepted, float alpha, float L, float U, float previous):
    cdef float target, minSoFar

    target = minimize_scalar(owtMinimization, bounds = (0,1), args=(carbon, solar, alpha, U, L, accepted), method='bounded').x
    return max(target, 0.0) 

cpdef float owtThreshold(float w, float U, float L, float alpha):
    return U + (U / alpha - U) * np.exp( w / alpha )

cpdef float owtMinimization(float x, float carbon, float solar, float alpha, float U, float L, float accepted):
    cdef float val
    val = solarOffset(carbon, solar, x)
    return (val) - integrate.quad(owtThreshold, accepted, (accepted + x), args=(U,L,alpha))[0]

# RORO algorithm implementation

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] roroOnline(list vals, list solarVals, float L, float U, float beta, float delivery):
    cdef int i
    cdef list sol
    cdef float cost, carbon, alpha, accepted, val, solar

    sol = []
    accepted = 0.0
    total = delivery/19

    # get value for alpha
    alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= 1:
            sol.append(0)
            continue

        solar = solarVals[i]
        
        remainder = (1 - accepted) * total
        
        if i == len(vals) - int(np.ceil(remainder)): # must accept last price(s)
            amount = min(remainder, 1)
            val = solarOffset(carbon, solar, amount)
            sol.append(amount)
            accepted += amount/total
            j = 1
            remainder = (1 - accepted) * total
            while remainder > 0:
                amount = min(remainder, 1)
                carbon = vals[i+j]
                solar = solarVals[i+j]
                val = solarOffset(carbon, solar, amount)
                sol.append(amount)
                accepted += amount/total
                j += 1
                remainder = (1 - accepted) * total
            break

        # solve for threshold-defined amount
        previous = 0
        if i != 0:
            previous = sol[-1]
        amount = roroHelper(carbon, solar, accepted, alpha, L, U, beta, previous, total)
        amount = min(total * amount, 1)
        accepted += amount/total
        val = solarOffset(carbon, solar, amount)
        sol.append(amount)

    cost = objectiveFunction(sol, vals, solarVals, beta)
    return sol, cost

# helper for RORO algorithm
cpdef float roroHelper(float carbon, float solar, float accepted, float alpha, float L, float U, float beta, float previous, float total):
    cdef float target

    try:
        target = minimize_scalar(roroMinimization, bounds = (0,1-accepted), args=(carbon, solar, alpha, U, L, beta, previous, accepted, total), method='bounded').x
        # solve for the amount
    except:
        print("something went wrong here w_j={}".format(accepted))
        return 0
    else:
        return max(target, 0.0)

cpdef float thresholdFunc(float w, float U, float L, float beta, float alpha):
    return U - beta + (U / alpha - U + 2 * beta) * np.exp( w / alpha )

cpdef float roroMinimization(float x, float carbon, float solar, float alpha, float U, float L, float beta, float previous, float accepted, float total):
    cdef float val
    val = solarOffset(carbon, solar, x)
    return (val) + ( beta * abs(x - (previous*1/total)) ) - integrate.quad(thresholdFunc, accepted, (accepted + x), args=(U,L,beta,alpha))[0]

cpdef tuple[list, float] convexComb(list vals, list solarVals, float beta, float lamda, list decision1, list decision2):
    cdef list sol
    sol = []
    for i in range(0, len(decision2)):
        sol.append(lamda * decision1[i] + (1-lamda) * decision2[i])
    
    cost = objectiveFunction(sol, vals, solarVals, beta)
    return sol, cost


cpdef tuple[list, float] carbonAgnostic(list vals, list solarVals, float beta, float delivery):
    cdef list sol
    cdef int index
    cdef float amount, total
    index = random.randint(0, len(vals)-1)
    sol = [0 for _ in range(len(vals))]
    total = delivery/19
    if total > 1:
        spots = int(np.ceil(total))
        topk = random.sample(range(0, len(vals)), spots)
        for k in topk:
            if total>1:
                sol[k] = 1
                total -= 1
            else:
                sol[k] = total
                total = 0
    else:
        sol[index] = total
    
    cost = objectiveFunction(sol, vals, solarVals, beta)
    return sol, cost