import numpy as np
from lib.MDP import MDP, MDPBase
from DP_algorithms import DPAlgorithms

class ModelBased(MDPBase):

    def __init__(self, mdp, sampleReward):
        super(ModelBased, self).__init__(mdp, sampleReward)

    def train(self, s0, defaultT, initialR, nEpisodes, nSteps, epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs:
        V -- final value function
        policy -- final policy
        '''

        # set all variables
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates, int)
        T = np.copy(defaultT)
        R = np.copy(initialR)
        visit_count = np.zeros((self.mdp.nActions, self.mdp.nStates))
        visit_count_transiton = np.zeros((self.mdp.nActions, self.mdp.nStates, self.mdp.nStates))
        episode_rewards = []

        for episode in range(nEpisodes):
            state = s0
            episode_rewards.append(0.0)
            for step in range(nSteps):
                # epsilon greedy select action
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.mdp.nActions)
                else:
                    action = policy[state]
                reward, next_state = self.sampleRewardAndNextState(state, action)
                episode_rewards[-1] += reward

                # get counts
                visit_count[action, state] += 1
                visit_count_transiton[action, state, next_state] += 1

                # update model
                T[action, state, :] = visit_count_transiton[action, state, :] / visit_count[action, state]
                R[action, state] = (reward + ((visit_count[action, state] - 1) * R[action, state])) / visit_count[
                    action, state]

                # Solve for V*
                mdp_model = MDP(T, R, self.mdp.discount)
                dpalgos = DPAlgorithms(mdp_model)
                [policy, V, iterId, tolerance] = dpalgos.modifiedPolicyIteration(policy, V, 10)

                state = next_state

        return [V, policy, np.array(episode_rewards)]