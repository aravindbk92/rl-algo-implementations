import numpy as np
from lib.MDP import MDPMaze
from DP_algorithms import DPAlgorithms

mdp = MDPMaze()
dpalgos = DPAlgorithms(mdp)
initial_policy = np.zeros(mdp.nStates,dtype=int)
initial_V = np.zeros(mdp.nStates)

print ("-----Value Iteration-----")
[V,nIterations,epsilon] = dpalgos.valueIteration(initialV=initial_V)
policy = dpalgos.extractPolicy(V)
print("V:")
print (V, "\n")
print ("Policy:")
print (policy, "\n")
print ("Number of iterations:", nIterations, "\n")

print ("-----Policy Iteration-----")
[policy,V,iterId] = dpalgos.policyIteration(initial_policy)
print("V:")
print (V, "\n")
print ("Policy:")
print (policy, "\n")
print ("Number of iterations:", iterId, "\n")

print ("-----Modified Policy Iteration-----")
nIterations_list = []
for partial_eval_iter in range(1,11):
   print (">>Partial eval iterations:", partial_eval_iter)
   [policy,V,iterId,tolerance] = dpalgos.modifiedPolicyIteration(initial_policy,initial_V,partial_eval_iter)
   print("V:")
   print (V, "\n")
   print ("Policy:")
   print (policy, "\n")
   print ("Number of iterations:", iterId, "\n")
   nIterations_list.append(iterId)
    
    

