# Time Fairness in Online Knapsack Problems
# Data Loading Helper (loads cloud traces in the form of MATLAB .mat files)

from mat4py import loadmat
import knapsack as k
import numpy as np
import seaborn as sns
import random
import pickle
import os
from math import e
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool

# function to load the specified trace with the specified value for theta
def loadFromMAT(theta=10):
    # load cloud traces
    traceValues = []
    traceWeights = []
    for i in range(1, 87):
        # loads a single trace from a single MAT file
        data = loadmat('data-cloud/values{}/jobvalue{}.mat'.format(theta, i))
        weights = data['jobweightCell']
        values = data['jobvalueCell']
        
        # flatten weights and values into a single list each
        weightsF = [x[0] for x in weights]
        valuesF = [x[0] for x in values]
        
        # append to running list
        traceValues.append(valuesF)
        traceWeights.append(weightsF)

    # compute L and U based on jobs in ALL traces
    ratios = []
    for (tValue, tWeight) in zip(traceValues, traceWeights):
        for weight,val in zip(tWeight, tValue):
            ratios.append(val/weight)
    L = min(ratios)
    U = max(ratios)
    # print(L)
    # print(U)

    W = 100

    # for each trace, shuffle it 100 times and add each to a list of artificial traces
    # also, compute the optimal solution to save computations, since order doesnt matter
    # also, compute d^star for each of the original traces (see Prediction Model in the paper), and save to bestDensities
    arTraceValues = []
    arTraceWeights = []
    optimalSols = []
    bestDensities = []
    for (tValue, tWeight) in zip(traceValues, traceWeights):
        # compute optimal solution
        weightsP = [int(x*100) for x in tWeight] # convert weights to integers for DP solution
        sol, packed = k.dpOptimalKnapsack(W, weightsP, tValue, len(tValue))
        packedDensities = [tValue[i]/tWeight[i] for i in packed]

        # find best constant threshold value density (i.e. d^star as defined in the paper)
        print(".", end="", flush=True)
        delta = 1
        minD = min(packedDensities)
        maxD = max(packedDensities)
        densities = [minD+(delta*i) for i in range(int((maxD-minD)/delta)+1)]
        sols = []
        ratios = []
        for density in densities: # try several possible values for d^star
            thresol = k.LAECT(1, tWeight, tValue, len(tValue), L, U, density, 1)[0][-1]
            sols.append(thresol)
            if thresol == 0:
                ratios.append(1000000)
                continue
            ratios.append(sol/thresol)
        ratios = np.array(ratios)
        bestDensity = densities[np.argmin(ratios)]

        # shuffle original trace 100 times
        for _ in range(100):
            tempValue = tValue.copy()
            tempWeight = tWeight.copy()

            # combine values and weights, then shuffle
            combined = list(zip(tempValue, tempWeight))
            random.shuffle(combined)
            shufValue, shufWeight = zip(*combined)

            # save shuffled traces, optimal solutions, and d^star value densities
            arTraceValues.append(shufValue)
            arTraceWeights.append(shufWeight)
            optimalSols.append(sol)
            bestDensities.append(bestDensity)
            
    # save the loaded traces to pickle files to speed things up next time
    os.makedirs("pickled{}".format(theta), mode = 0o777, exist_ok = True)
    pickle.dump( arTraceValues, open( "pickled{}/traces_values.pickle".format(theta), "wb" ) )
    pickle.dump( arTraceWeights, open( "pickled{}/traces_weights.pickle".format(theta), "wb" ) )
    pickle.dump( optimalSols, open( "pickled{}/optimal_sols.pickle".format(theta), "wb" ) )
    pickle.dump( bestDensities, open( "pickled{}/optimal_dens.pickle".format(theta), "wb" ) )
    pickle.dump( (L, U), open( "pickled{}/bounds.pickle".format(theta), "wb" ) )

    return arTraceValues, arTraceWeights, L, U, optimalSols, bestDensities

def loadDataAndOPT(theta=10):
    # try to load from pickle file
    try:
        arTraceValues = pickle.load( open( "pickled{}/traces_values.pickle".format(theta), "rb" ) )
        arTraceWeights = pickle.load( open( "pickled{}/traces_weights.pickle".format(theta), "rb" ) )
        optimalSols = pickle.load( open( "pickled{}/optimal_sols.pickle".format(theta), "rb" ) )
        bestDensities = pickle.load( open( "pickled{}/optimal_dens.pickle".format(theta), "rb" ) )
        L, U = pickle.load( open( "pickled{}/bounds.pickle".format(theta), "rb" ) )
        print("Loaded traces from pickle file")
    except (OSError, IOError) as e:
        print("No pickle file found. Loading traces from MATLAB files...")
        arTraceValues, arTraceWeights, L, U, optimalSols, bestDensities = loadFromMAT(theta=theta)
    return arTraceValues, arTraceWeights, L, U, optimalSols, bestDensities