import numpy as np
from lib.MDP import MDP, MDPMaze
from model_based import ModelBased

# Test ModelBased RL with simple MDP
print ("\nModelBased RL results")
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP(T,R,discount)
initial = np.random.rand(mdp.nActions,mdp.nStates)

model_based_rl = ModelBased(mdp,np.random.normal)
[V, policy, episode_rewards] = model_based_rl.train(s0=0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,initialR=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)
print (policy)

# Test ModelBased RL with maze MDP
print ("\nModelBased RL Maze results")
maze_mdp = MDPMaze()
initial = np.random.rand(maze_mdp.nActions,maze_mdp.nStates)

model_based_rl = ModelBased(maze_mdp,np.random.normal)
[V, policy, episode_rewards] = model_based_rl.train(s0=0,defaultT=np.ones([maze_mdp.nActions,maze_mdp.nStates,maze_mdp.nStates])/maze_mdp.nStates,initialR=np.zeros([maze_mdp.nActions,maze_mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.05)
print (policy)