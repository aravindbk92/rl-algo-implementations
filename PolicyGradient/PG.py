import numpy as np

class RLPolicyGradient:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs:
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def sampleSoftmaxPolicy(self, policyParams, state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs:
        action -- sampled action
        '''
        all_actions = list(range(self.mdp.nActions))
        softmax = self.softmax(np.array(policyParams[:, state]))

        # sample an action from probabilites
        action = np.random.choice(all_actions, p=softmax)

        return action

    def reinforce(self, s0, initialPolicyParams, nEpisodes, nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs:
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # set variables
        policyParams = np.copy(initialPolicyParams)
        visit_count = np.zeros((self.mdp.nActions, self.mdp.nStates))
        episode_rewards = []

        for episodes in range(nEpisodes):
            episode_rewards.append(0.0)
            episode = np.empty((0, 3))
            state = s0

            # generate episode
            for n in range(nSteps):
                # perform an action and get reward
                action = self.sampleSoftmaxPolicy(policyParams, state)
                visit_count[action, state] += 1
                reward, next_state = self.sampleRewardAndNextState(state, action)
                episode_rewards[-1] += reward

                # store current state to epsiode
                episode = np.vstack((episode, [state, action, reward]))

                state = next_state

            # train policy using genrated episode
            alpha = 0.01
            for n in range(nSteps):
                # get current state, action and rreward
                rewards = episode[n:, 2]
                state = int(episode[n, 0])
                action = int(episode[n, 1])

                # calculate G
                discounts = np.power(self.mdp.discount, np.linspace(0, nSteps - n - 1, nSteps - n))
                G_n = np.sum(discounts * rewards)

                # Calculate gradient
                gradient = self.logsoftmax_grad(policyParams, state, action)

                # Update policy params
                updatedParams = policyParams[:, state] + alpha * self.mdp.discount ** n * G_n * gradient
                policyParams.T[state] = updatedParams

        return [policyParams, np.array(episode_rewards)]

    def softmax(self, x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def logsoftmax_grad(self, policyParams, state, action):
        softmax = self.softmax(np.array(policyParams[:, state]))
        softmax = -softmax
        softmax[action] = 1 - softmax[action]
        return softmax

    def get_policy(self, policy_params):
        policy_action = []
        for s in range(self.mdp.nStates):
            softmax = self.softmax(np.array(policy_params[:, s]))
            policy_action.append(np.argmax(softmax))
        return policy_action