# Time Fairness in Online Knapsack Problems
# Algorithm Implementations

import numpy as np
import random
import math
from math import e
from math import log
import matplotlib.pyplot as plt
import scipy

# knapsack of capacity W        -- W
# list of weights for each item -- weights
# list of values for each item  -- vals
# number of items               -- n
def dpOptimalKnapsack(W, weights, vals, n):
    dp = [0 for i in range(W + 1)]  # Making the dp array
    packed = set()

    if W < min(weights):
        return 0, set()

    for i in range(1, n + 1):  # taking first i elements
        for w in range(W, 0, -1):  # starting from back,so that we also have data of
            # previous computation when taking i-1 items
            if weights[i - 1] <= w:
                if dp[w - weights[i - 1]] + vals[i - 1] > dp[w]:
                    packed.add(i-1)
                else:
                    if (i - 1) in packed:
                        packed.remove(i-1)
                dp[w] = max(dp[w], dp[w - weights[i - 1]] + vals[i - 1])

    return dp[W], packed  # returning the maximum value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
def ZCL(W, weights, vals, n, L, U):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        z_j = (W - remainingW) / W  # how much of knapsack is occupied
        phi_j = phi(z_j, L, U)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
def ZCLRandomized(W, weights, vals, n, L, U):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []
    z_j = random.uniform(0, 1)  # generate threshold from phi threshold function

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        phi_j = phi(z_j, L, U)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
# fairness parameter \in [0,1]          -- alpha
def baseline(W, weights, vals, n, L, U, alpha):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        z_j = (W - remainingW) / W  # how much of knapsack is occupied
        phi_j = alphaPhi(z_j, L, U, alpha)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
# fairness parameter \in [0,1]          -- alpha
def ECT(W, weights, vals, n, L, U, alpha):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        z_j = (W - remainingW) / W  # how much of knapsack is occupied
        psi_j = alphaFair(z_j, L, U, alpha)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= psi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
# fairness parameter \in [0,1]          -- alpha
def LAECT(W, weights, vals, n, L, U, hat_d, gamma):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        z_j = (W - remainingW) / W  # how much of knapsack is occupied
        psi_j = alphaLA(z_j, L, U, hat_d, gamma)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= psi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# helper function phi for ZCL algos
def phi(z, L, U):
    return (((U*e)/L)**z)*(L/e)

# helper function phi for baseline algo
def alphaPhi(z, L, U, alpha):
    ell = ((alpha * log(U/L)) + alpha - 1) / (log(U/L))
    if z < alpha:
        return L
    else:
        return (((U*e)/L)**((z-ell)/(1-ell)))*(L/e)

# helper function phi for ECT algo
def alphaFair(z, L, U, alpha):
    beta = (scipy.special.lambertw(((U-U*alpha)/(L*alpha)), k=0))/(1-alpha)
    if z < alpha:
        return L
    else:
        return (U*(e**(beta*(z-1))))
    
# helper function phi for LAECT algo
def alphaLA(z, L, U, hat_d, gamma):
    if gamma == 1:
        return hat_d
    exp = (((U*e)/L)**(z/(1-gamma)))*(L/e)
    if exp < hat_d:
        return exp
    elif exp >= hat_d:
        exp2 = (((U*e)/L)**((z-gamma)/(1-gamma)))*(L/e)
        if exp2 >= hat_d:
            return exp2
        else:
            return hat_d