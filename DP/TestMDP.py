import numpy as np
from lib.MDP import MDP
from DP_algorithms import DPAlgorithms
import matplotlib.pyplot as plt

''' Construct simple MDP '''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)
dpalgos = DPAlgorithms(mdp)

'''Test each procedure'''
print ("-----Value Iteration-----")
[V,nIterations,epsilon] = dpalgos.valueIteration(initialV=np.zeros(mdp.nStates))
policy = dpalgos.extractPolicy(V)
print("V:")
print (V, "\n")
print ("Policy:")
print (policy, "\n")
print ("Number of iterations:", nIterations, "\n")

print ("-----Policy Iteration-----")
[policy,V,iterId] = dpalgos.policyIteration(np.array([0,0,0,0]))
print("V:")
print (V, "\n")
print ("Policy:")
print (policy, "\n")
print ("Number of iterations:", iterId, "\n")

print ("-----Modified Policy Iteration-----")
nIterations_list = []
for partial_eval_iter in range(1,11):
    print (">>Partial eval iterations:", partial_eval_iter)
    [policy,V,iterId,tolerance] = dpalgos.modifiedPolicyIteration(np.array([0,0,0,0]),np.array([0,0,0,0]),partial_eval_iter)
    print("V:")
    print (V, "\n")
    print ("Policy:")
    print (policy, "\n")
    print ("Number of iterations:", iterId, "\n")
    nIterations_list.append(iterId)

plt.xlabel("Iterations in Partial Policy Evaluation")
plt.ylabel("Number of iterations to converge")
plt.plot(nIterations_list)
plt.savefig("niters_partial_policy_eval.png")



