import numpy as np
from lib.MDP import MDP, MDPMaze
from PG import RLPolicyGradient

# Test Q-learning with simple MDP
print ("\nREINFORCE results")
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP(T,R,discount)
initial = np.random.rand(mdp.nActions,mdp.nStates)

rl_instance = RLPolicyGradient(mdp,np.random.normal)
updated_policy_params, episode_rewards = rl_instance.reinforce(s0=0,initialPolicyParams=initial,nEpisodes=1000,nSteps=100)
print (rl_instance.extract_policy(updated_policy_params))

# Test REINFORCE with maze MDP
print ("\nREINFORCE Maze results")
maze_mdp = MDPMaze()
initial = np.random.rand(maze_mdp.nActions,maze_mdp.nStates)

rl_instance = RLPolicyGradient(maze_mdp,np.random.normal)
policy_params, episode_rewards = rl_instance.reinforce(s0=0,initialPolicyParams=initial,nEpisodes=1000,nSteps=100)
policy = rl_instance.extract_policy(policy_params)
print (policy)