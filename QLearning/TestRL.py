import numpy as np
from lib.MDP import MDP, MDPMaze
from RL import RL

# Test Q-learning with simple MDP
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP(T,R,discount)
rl_instance = RL(mdp,np.random.normal)

[Q,policy,episode_rewards] = rl_instance.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=1000,nSteps=100,epsilon=0.3)
print ("\nQ-learning results")
print (Q)
print (policy)

# Test Q-learning with maze MDP
maze_mdp = MDPMaze()
rl_instance = RL(maze_mdp,np.random.normal)
[Q, policy, episode_rewards] = rl_instance.qLearning(s0=0, initialQ=np.zeros([mdp.nActions, mdp.nStates]), nEpisodes=200,
                                                   nSteps=100, epsilon=0.3)
print ("\nQ-learning results")
print (Q)
print (policy)