# Time Fairness in Online Knapsack Problems
# Experiments

import knapsack as k
import numpy as np
import seaborn as sns
import random
import pickle
from math import e
import math
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool

import matplotlib.style as style
style.use('seaborn-colorblind')


# function stubs for multiprocessing
def OPT_unpack(args):
    return k.dpOptimalKnapsack(*args)

def ZCL_unpack(args):
    return k.ZCL(*args)[0][-1] # profit is returned over time, so just get the final profit

def ZCLRandomized_unpack(args):
    return k.ZCLRandomized(*args)[0][-1] # profit is returned over time, so just get the final profit

def ECT_unpack(args):
    return k.ECT(*args)[0][-1] # profit is returned over time, so just get the final profit

def LAECT_unpack(args): 
    return k.LAECT(*args)[0][-1]  # profit is returned over time, so just get the final profit

# main functions for experiments.
# error is a parameter controlling how much multiplicative error to add to the predictions
def experimentLA(error):
    import load_traces

    # get traces, L, U, optimal solutions, and bestDens, where each density is the d^star value for the corresponding trace
    tracesVal, tracesWgt, L, U, optimalSols, bestDens = load_traces.loadDataAndOPT(theta=50)

    # let's do some experiments!

    # set knapsack capacity
    W = 1
    lengths = [len(x) for x in tracesVal]

    # add some gaussian (mean 0) noise to the predictions, with stddev = error
    predictions = [x*abs(1 + random.gauss(0, error)) for x in bestDens]

    # compute ZCL, ECT, and LA-ECT solutions for each trace using a thread pool
    ZCLSols = []
    with Pool(10) as p:
        ZCLSols = p.map(ZCL_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U)))

    ECTSols = []
    with Pool(10) as p:
        ECTSols = p.map(ECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), itertools.repeat(0.5)))

    LAECT1Sols = []
    with Pool(10) as p:
        LAECT1Sols = p.map(LAECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), predictions, itertools.repeat(0.33)))

    LAECT2Sols = []
    with Pool(10) as p:
        LAECT2Sols = p.map(LAECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), predictions, itertools.repeat(0.66)))

    LAECT3Sols = []
    with Pool(10) as p:
        LAECT3Sols = p.map(LAECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), predictions, itertools.repeat(1)))

    # compute empirical competitive ratios for each set of solutions, using numpy
    # convert optimal sols to numpy array
    optimalSols = np.array(optimalSols)
    ZCLSols = np.array(ZCLSols)
    ECTSols = np.array(ECTSols)
    LAECT1Sols = np.array(LAECT1Sols)
    LAECT2Sols = np.array(LAECT2Sols)
    LAECT3Sols = np.array(LAECT3Sols)

    # compute all empirical rattios (optimalSols / algSols, etc.)
    ZCLRatios = optimalSols/ZCLSols
    ECTRatios = optimalSols/ECTSols
    LAECT1Ratios = optimalSols/LAECT1Sols
    LAECT2Ratios = optimalSols/LAECT2Sols
    LAECT3Ratios = optimalSols/LAECT3Sols

    # print average competitive ratios to console
    print("competitive ratios: ")
    print("ZCL: {}".format(np.mean(ZCLRatios)))
    print("ECT[0.5]: {}".format(np.mean(ECTRatios)))
    print("LA-ECT[0.33]: {}".format(np.mean(LAECT1Ratios)))
    print("LA-ECT[0.66]: {}".format(np.mean(LAECT2Ratios)))
    print("LA-ECT[1]: {}".format(np.mean(LAECT3Ratios)))

    # return values for plotting
    return ZCLRatios, ECTRatios, LAECT1Ratios, LAECT2Ratios, LAECT3Ratios
 

if __name__ == "__main__":
    # for each prediction error value (refer to experiments section in the paper), load data and run experiments
    for i, error in enumerate([0, (0.5), (1)]):
        ZCLRatios, ECTRatios, LAECT1Ratios, LAECT2Ratios, LAECT3Ratios = experimentLA(error=error)

        # set plot size to (4, 3) and dpi = 500
        plt.figure(figsize=(4, 3), dpi=500)

        linestyles = ['-', ':', '--', '--', '--', '--', '--']

        # CDF plot for competitive ratio (across all experiments)
        legend = ["ZCL", "ECT[α = 0.5]", "LA-ECT[γ = 0.33]", "LA-ECT[γ = 0.66]", "LA-ECT[γ = 1]"]
        for j, (dat,ls) in enumerate(zip([ZCLRatios, ECTRatios, LAECT1Ratios, LAECT2Ratios, LAECT3Ratios], linestyles)):
            if j < 4:
                sns.ecdfplot(data = dat, linestyle = ls) # plots empirical CDF
            else:
                sns.ecdfplot(data = dat, linestyle = ls, color="C5") # plots empirical CDF (skips a bad color)

        # set labels and limits
        plt.ylabel('empirical CDF')
        plt.xlabel("empirical competitive ratio")
        plt.xlim(0.9, 12)

        # [ this can be skipped if you don't need a legend ]
        # change legend to be located outside the plot (bottom center)
        # plt.legend(legend, ncol=5, loc='lower center', bbox_to_anchor=(0.5, -0.3))

        # save plot to file
        plt.tight_layout()
        plt.savefig("error{}.png".format(i))
        plt.clf()

