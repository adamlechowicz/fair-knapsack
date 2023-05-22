# Time Fairness in Online Knapsack Problems
# Experiments

import knapsack as k
import numpy as np
import seaborn as sns
import random
import pickle
from math import e
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool

import matplotlib.style as style
style.use('tableau-colorblind10')


# function stubs for multiprocessing
def OPT_unpack(args):
    return k.dpOptimalKnapsack(*args)

def ZCL_unpack(args):
    return k.ZCL(*args)[0][-1] # profit is returned over time, so just get the final profit

def ZCLRandomized_unpack(args):
    return k.ZCLRandomized(*args)[0][-1] # profit is returned over time, so just get the final profit

def ECT_unpack(args):
    return k.ECTNew(*args)[0][-1] # profit is returned over time, so just get the final profit

def baseline_unpack(args):
    return k.baseline(*args)[0][-1] # profit is returned over time, so just get the final profit


# main functions for experiments.
# theta is a parameter controlling which of the three data sets to use (see load_traces.py)
def experiment(theta):
    import load_traces

    # get traces, L, U, and optimal solutions
    tracesVal, tracesWgt, L, U, optimalSols, bestDens = load_traces.loadDataAndOPT(theta=theta)

    # let's do some experiments!

    # set knapsack capacity to 1, compute lengths of the traces
    W = 1
    lengths = [len(x) for x in tracesVal]

    # compute ZCL, ZCLRandomized, ECT, and baseline algorithm solutions for each trace, using a thread pool
    ZCLSols = []
    with Pool(10) as p:
        ZCLSols = p.map(ZCL_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U)))

    ZCLRandomizedSols = []
    with Pool(10) as p:
        ZCLRandomizedSols = p.map(ZCLRandomized_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U)))

    ECT1Sols = [] # alpha = 0.33
    with Pool(10) as p:
        ECT1Sols = p.map(ECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), itertools.repeat(0.33)))

    ECT2Sols = [] # alpha = 0.66
    with Pool(10) as p:
        ECT2Sols = p.map(ECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), itertools.repeat(0.66)))

    baselineSols = [] # alpha = 0.66
    with Pool(10) as p:
        baselineSols = p.map(baseline_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), itertools.repeat(0.66)))

    ECT3Sols = [] # alpha = 1
    with Pool(10) as p:
        ECT3Sols = p.map(ECT_unpack, zip(itertools.repeat(W), tracesWgt, tracesVal, lengths, itertools.repeat(L), itertools.repeat(U), itertools.repeat(1)))

    # compute empirical competitive ratios for each set of solutions, using numpy
    # convert optimal sols to numpy array
    optimalSols = np.array(optimalSols)
    ZCLSols = np.array(ZCLSols)
    ZCLRandomizedSols = np.array(ZCLRandomizedSols)
    ECT1Sols = np.array(ECT1Sols)
    ECT2Sols = np.array(ECT2Sols)
    baselineSols = np.array(baselineSols)
    ECT3Sols = np.array(ECT3Sols)

    # compute all empirical rattios (optimalSols / algSols, etc.)
    ZCLRatios = optimalSols/ZCLSols
    ZCLRandomizedRatios = optimalSols/ZCLRandomizedSols
    ECT1Ratios = optimalSols/ECT1Sols
    ECT2Ratios = optimalSols/ECT2Sols
    baselineRatios = optimalSols/baselineSols
    ECT3Ratios = optimalSols/ECT3Sols

    # print average competitive ratios to console
    print("competitive ratios: ")
    print("ZCL: {}".format(np.mean(ZCLRatios)))
    print("ZCLRandomized: {}".format(np.mean(ZCLRandomizedRatios)))
    print("ECT[0.33]: {}".format(np.mean(ECT1Ratios)))
    print("ECT[0.66]: {}".format(np.mean(ECT2Ratios)))
    print("Baseline[0.66]: {}".format(np.mean(baselineRatios)))
    print("ECT[1]: {}".format(np.mean(ECT3Ratios)))

    # return values for plotting
    return ZCLRatios, ZCLRandomizedRatios, ECT1Ratios, ECT2Ratios, baselineRatios, ECT3Ratios
 

if __name__ == "__main__":
    # for each value of theta, which corresponds to different values of U/L in [500, 2500, 12500], load the data set and run experiments
    for theta in [10, 50, 250]:
        ZCLRatios, ZCLRandomizedRatios, ECT1Ratios, ECT2Ratios, baselineRatios, ECT3Ratios = experiment(theta=theta)

        # set plot size to (4, 3) and dpi = 500
        plt.figure(figsize=(4, 3), dpi=500)

        linestyles = ['-', '--', ':', ':', '-.', ':', ':']

        # CDF plot for competitive ratio (across all experiments)
        legend = ["ZCL", "ZCLRandomized", "ECT[alpha = 0.25]", "ECT[alpha = 0.5]", "baseline algorithm [alpha = 0.5]", "ECT[alpha = 0.75]", "ECT[alpha = 1]"]
        for dat,ls in zip([ZCLRatios, ZCLRandomizedRatios, ECT1Ratios, ECT2Ratios, baselineRatios, ECT3Ratios], linestyles):
            sns.ecdfplot(data = dat, linestyle = ls) # plots empirical CDF

        # set labels and limits 
        plt.ylabel('empirical CDF')
        plt.xlabel("empirical competitive ratio")
        plt.xlim(0.9, 12)

        # [ this can be skipped if you don't need a legend ]
        # change legend to be located outside the plot (bottom center)
        # plt.legend(legend, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.3))

        # save plot to file
        plt.tight_layout()
        plt.savefig("theta{}.png".format(theta))
        plt.clf()

