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

# def randomInterval(df, T, noiseFactor=0):
#     randInd = random.randrange(len(df)-T)
#     int = df[randInd:randInd+T]
#     noise = np.random.normal(loc=0.0, scale=noiseFactor, size=T)
#     interval = int['carbon_intensity_avg'].values
#     interval = interval + noise
#     interval = interval.clip(min=0)
#     return interval.tolist()

# def addNoise(vals, noiseFactor):
#     vals = np.array(vals)
#     mean = np.mean(vals)
#     delta = vals - mean
#     # noise = np.random.normal(loc=0.0, scale=noiseFactor, size=len(vals))
#     noisyInterval = np.full_like(vals, mean) + (noiseFactor*delta)
#     noisyInterval = noisyInterval.clip(min=0)
#     return noisyInterval.tolist()

# def boundedValues(df):
#     L = df["carbon_intensity_avg"].min()
#     U = df["carbon_intensity_avg"].max()
#     return L, U

# list of values            -- vals
# length of subsequence     -- k
# def smallestSubsequenceK(vals, k):
#     subseq = []
#     indices = (0,0)
#     curSum = math.inf
#     n = len(vals)

#     for i in range(0, n-k+1):  
#         subarray = vals[i:i+k]
#         if sum(subarray) < curSum:
#             subseq = subarray
#             curSum = sum(subarray)
#             indices = (i, i+k)

#     return subseq, indices  # returning the min subsequence, and its indices


# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
# def dynProgOPT(vals, k, beta):
#     minCost = math.inf
#     sol = []
#     # n = len(vals)
#     if (k == 0):
#         return sol, 0

#     for i in range(1, k+1):
#         newVals = vals.copy()
#         subseq, indices = smallestSubsequenceK(newVals, i) # get the smallest subsequence of length i

#         # subtract subseq from vals
#         del newVals[indices[0]:indices[1]]

#         otherSeq, otherSum = dynProgOPT(newVals, k-i, beta)
#         curCost = sum(subseq) + otherSum + 2*beta

#         if curCost < minCost:
#             minCost = curCost
#             sol = []
#             sol.append(subseq)
#             for seq in otherSeq:
#                 sol.append(seq)
    
#     return sol, minCost

# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
# def carbonAgnostic(vals, k, beta):
#     subseq = vals[0:k]
#     cost = sum(subseq) + 2*beta
    
#     return [subseq], cost

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

cpdef tuple[list, float] optimalSolutionLin(list vals, float beta):
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
    b.append(-1)

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

    # print results
    # if results.status == 0: print(f'The solution is optimal.') 
    # print(f'Objective value: z* = {results.fun}')
    # print(f'Solution: x* = {results.x[0:n]}')
    # print(sum(results.x[0:n]))

    # return optimal variable settings + cost
    if results.status == 0:
        return results.x, results.fun
    else:
        print("Error: no optimal solution found.")
        results = minimize(objectiveFunction, x0=(np.zeros(n)), args=(vals, solarVals, beta), bounds=all_bounds, constraints=constraint)
        return results.x, results.fun
        return [], math.inf

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] oneMinOnline(list vals, list solarVals, int k, float L, float U, float beta):
    cdef int lastElem, i
    cdef bint accept, prevAccepted
    cdef list sol
    cdef float cost, threshold, val, accepted
    
    prevAccepted = False
    sol = []
    accepted = 0
    lastElem = len(vals)
    cost = 0

    threshold = math.sqrt(U*L)

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= k:
            sol.append(0)
            continue
        
        if accepted + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break

        solar = solarVals[i]
        val = solarOffset(carbon, solar, 1/k)
        
        accept = (val <= threshold)
        if prevAccepted != accept:
            cost += beta * (1/k)
        if accept:
            sol.append(1/k)
            accepted += 1
            cost += val * (1/k)
            if accepted == k:
                cost += beta * (1/k) # one last switching cost to turn off
                continue
        else:
            sol.append(0)
        prevAccepted = accept

    if accepted < k:
        if prevAccepted != True:
            cost += 2*beta * (1/k)
        for i in range(lastElem, len(vals)):
            solar = solarVals[i]
            carbon = vals[i]
            val = solarOffset(carbon, solar, 1/k)

            sol.append(1/k)
            cost += val * (1/k)

    return sol, cost


# standard one-way trading algorithm implementation

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] owtOnline(list vals, list solarVals, float L, float U, float beta):
    cdef int i
    cdef list sol
    cdef float cost, carbon, alpha, accepted, val, solar

    sol = []
    accepted = 0.0
    cost = 0.0

    # get value for alpha
    alpha = 1 / (1 + lambertw( ( (L/U) - 1 ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= 1:
            sol.append(0)
            continue
        
        solar = solarVals[i]
        
        if i == len(vals) - 1: # must accept last price
            remainder = 1.0 - accepted
            val = solarOffset(carbon, solar, remainder)
            cost += val # hitting cost
            cost += (beta * abs(remainder - sol[-1]))
            cost += (beta * abs(0 - remainder)) # one last switching cost to turn off
            sol.append(remainder)
            accepted += remainder
            break
        
        # solve for threshold-defined amount
        previous = 0
        if i != 0:
            previous = sol[-1]
        amount = owtHelper(carbon, solar, accepted, alpha, L, U, previous)
        val = solarOffset(carbon, solar, amount)
        accepted += amount
        cost += val
        cost += (beta * abs(amount - previous))
        sol.append(amount)
        if accepted >= 1:
            cost += (beta * abs(0 - amount)) # one last switching cost to turn off
            continue

    return sol, cost

# helper for one-way trading algorithm
cpdef float owtHelper(float carbon, float solar, float accepted, float alpha, float L, float U, float previous):
    cdef float target, minSoFar

    target = minimize_scalar(owtMinimization, bounds = (0,1), args=(carbon, solar, alpha, U, L, accepted), method='bounded').x
    return target

    # minSoFar = np.inf

    # for x in np.linspace(0, 1, 1000):
    #     if x > 1 - accepted:
    #         break
    #     cur = owtMinimization(x, carbon, solar, alpha, U, L, accepted)
    #     if cur < minSoFar:
    #         minSoFar = cur
    #         target = x
    #     if x > 0 and cur > 10:
    #         break
        
    # return target

    # try:
    #     target = minimize(owtMinimization, accepted, args=(val, alpha, U, L, previous, accepted), bounds = [(0,1)]).x[0]
    #     # solve for the amount
    # except:
    #     print("something went wrong here w_j={}".format(accepted))
    # else:
    #     return target

cpdef float owtThreshold(float w, float U, float L, float alpha):
    return U + (U / alpha - U) * np.exp( w / alpha )

cpdef float owtMinimization(float x, float carbon, float solar, float alpha, float U, float L, float accepted):
    cdef float val
    val = solarOffset(carbon, solar, x)
    return (val) - integrate.quad(owtThreshold, accepted, accepted + x, args=(U,L,alpha))[0]

# RORO algorithm implementation

# list of costs (values)    -- vals
# list of solar availability-- solarVals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] roroOnline(list vals, list solarVals, float L, float U, float beta):
    cdef int i
    cdef list sol
    cdef float cost, carbon, alpha, accepted, val, solar

    sol = []
    accepted = 0
    cost = 0

    # get value for alpha
    alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, carbon) in enumerate(vals):
        if accepted >= 1:
            sol.append(0)
            continue

        solar = solarVals[i]
        
        if i == len(vals) - 1: # must accept last price
            remainder = 1.0 - accepted
            val = solarOffset(carbon, solar, remainder)
            cost += val # hitting cost
            cost += (beta * abs(remainder - sol[-1]))
            cost += (beta * abs(0 - remainder)) # one last switching cost to turn off
            sol.append(remainder)
            accepted += remainder
            break
                
        # solve for threshold-defined amount
        previous = 0
        if i != 0:
            previous = sol[-1]
        amount = roroHelper(carbon, solar, accepted, alpha, L, U, beta, previous)
        val = solarOffset(carbon, solar, amount)
        accepted += amount
        cost += val
        cost += (beta * abs(amount - previous))
        sol.append(amount)
        if accepted >= 1:
            cost += (beta * abs(0 - amount)) # one last switching cost to turn off
            continue

    return sol, cost

# helper for RORO algorithm
cpdef float roroHelper(float carbon, float solar, float accepted, float alpha, float L, float U, float beta, float previous):
    cdef float target

    # target = minimize(roroMinimization, accepted, args=(val, alpha, U, L, beta, previous, accepted), bounds = [(0,1)]).x[0]
    # return target

    try:
        target = minimize_scalar(roroMinimization, bounds = (0,1-accepted), args=(carbon, solar, alpha, U, L, beta, previous, accepted), method='bounded').x
        # solve for the amount
    except:
        print("something went wrong here w_j={}".format(accepted))
        return 0
    else:
        return target

cpdef float thresholdFunc(float w, float U, float L, float beta, float alpha):
    return U - beta + (U / alpha - U + 2 * beta) * np.exp( w / alpha )

cpdef float roroMinimization(float x, float carbon, float solar, float alpha, float U, float L, float beta, float previous, float accepted):
    val = solarOffset(carbon, solar, x)
    return (val) + ( beta * abs(x - previous) ) - integrate.quad(thresholdFunc, accepted, accepted + x, args=(U,L,beta,alpha))[0]

cpdef float thresholdOLDFunc(float w, float U, float L, float beta, float alpha):
    return U - beta + (U / alpha - U + 2 * beta) * np.exp( w / alpha )

cpdef float roroOLDMinimization(float x, float val, float alpha, float U, float L, float beta, float previous, float accepted):
    return (x * val) + ( beta * abs(x - previous) ) - integrate.quad(thresholdOLDFunc, accepted, accepted + x, args=(U,L,beta,alpha))[0]


# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] roroOLDOnline(list vals, float L, float U, float beta):
    cdef int i
    cdef list sol
    cdef float cost, val, alpha, accepted, adjustedVal

    sol = []
    accepted = 0
    cost = 0

    # get value for alpha
    alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if accepted >= 1:
            sol.append(0)
            continue
        
        if i == len(vals) - 1: # must accept last price
            remainder = 1.0 - accepted
            cost += val * remainder # hitting cost
            cost += (beta * abs(remainder - sol[-1]))
            cost += (beta * abs(0 - remainder)) # one last switching cost to turn off
            sol.append(remainder)
            accepted += remainder
            break
                
        # solve for threshold-defined amount
        previous = 0
        if i != 0:
            previous = sol[-1]
        amount = roroOLDHelper(val, accepted, alpha, L, U, beta, previous)
        accepted += amount
        cost += val * amount
        cost += (beta * abs(amount - previous))
        sol.append(amount)
        if accepted >= 1:
            cost += (beta * abs(0 - amount)) # one last switching cost to turn off
            continue

    return sol, cost

# helper for RORO algorithm
cpdef float roroOLDHelper(float val, float accepted, float alpha, float L, float U, float beta, float previous):
    cdef float target, minSoFar

    # target = minimize_scalar(roroOLDMinimization, bounds = (0,1-accepted), args=(val, alpha, U, L, beta, previous, accepted), method = 'bounded', options={'maxiter': 50000}).x
    # return target

    minSoFar = np.inf
    for x in np.linspace(0, 1, 1000):
        if x > 1 - accepted:
            break
        cur = roroOLDMinimization(x, val, alpha, U, L, beta, previous, accepted)
        if cur < minSoFar:
            minSoFar = cur
            target = x
        if x > 0 and cur > 10:
            break
        
    return target

    # try:
    #     target = minimize_scalar(roroOLDMinimization, bounds = (0,1-accepted), args=(val, alpha, U, L, beta, previous, accepted), method='bounded').x
    #     # solve for the amount
    # except:
    #     print("something went wrong here w_j={}".format(accepted))
    #     return 0
    # else:
    #     return target




# need to fix this.