import random
import math
from scipy.optimize import linprog
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

# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
cpdef tuple[list, float] oneMinOnline(list vals, int k, float U, float L, float beta):
    cdef int accepted, lastElem, i
    cdef bint accept, prevAccepted
    cdef list sol, runningList
    cdef float cost, threshold, val

    prevAccepted = False
    sol = []
    accepted = 0
    runningList = []
    lastElem = len(vals)
    cost = 0
    L = 35
    U = 1105

    threshold = math.sqrt(U*L)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if accepted + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = (val <= threshold)
        if prevAccepted != accept:
            if len(runningList) > 1:
                sol.append(runningList)
            runningList = []
            cost += beta
        if accept:
            runningList.append(val)
            accepted += 1
            cost += val
            if accepted == k:
                sol.append(runningList)
                cost += beta # one last switching cost to turn off
                break
        prevAccepted = accept

    if accepted < k:
        if prevAccepted != True:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            runningList.append(vals[i])
            cost += vals[i]
        sol.append(runningList)

    return sol, cost


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


cpdef tuple[list, float] optimalSol(list vals, float beta):
    cdef int n, dim, i
    cdef list c, A, b, row, all_bounds
    cdef float val

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
    row[i] = 1
    row[i+n] = -1
    A.append(row)
    b.append(0)
    for i in range(0, n):
        row = [0 for i in range(0, dim)]
        row[i] = -1
        row[i+1] = 1
        row[i+n] = -1
        A.append(row.copy())
        b.append(0)

        row[i] = 1
        row[i+1] = -1
        A.append(row.copy())
        b.append(0)
    row = [0 for i in range(0, dim)]
    row[n-1] = 1
    row[-1] = -1
    A.append(row)
    b.append(0)

    print(A)
    print(b)

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