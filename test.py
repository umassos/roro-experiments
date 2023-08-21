import time

import pyximport
pyximport.install()

import functions as f

if __name__ == '__main__':
    # generate a synthetic sequence
    seq = f.generateSyntheticSequence(10, 30, 1)
    print(seq)

    # set the switching cost
    beta = 20

    # get the optimal solution
    opt, optCost = f.optimalSolution(seq, beta)

    # get the one min solution
    onemin, oneminCost = f.oneMinOnline(seq, 1, min(seq), max(seq), beta)

    # get the one way trading solution
    owt, owtCost = f.owtOnline(seq, min(seq), max(seq), beta)

    # get the RORO solution
    roro, roroCost = f.roroOnline(seq, min(seq), max(seq), beta)

    # print the sequence
    print("Sequence: ", seq)

    # print the optimal solution and cost
    print("OPT: ", opt[0:len(seq)], " OPT Cost: ", optCost)
    # print("OPT switching: ", opt[len(seq):])

    # print the one min solution and cost
    print("1-min: ", onemin, " 1-min Cost: ", oneminCost)

    # print the one way trading solution and cost
    print("OWT: ", owt, " OWT Cost: ", owtCost)

    # print the RORO solution and cost
    print("RORO: ", roro, " RORO Cost: ", roroCost)


