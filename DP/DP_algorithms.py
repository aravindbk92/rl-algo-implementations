import numpy as np
from lib.MDP import MDP
from lib.MDP import MDPBase

class DPAlgorithms(MDPBase):

    def __init__(self, mdp):
        super(DPAlgorithms, self).__init__(mdp)

    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
         
        V_old = initialV
        iterId = 1
        while iterId <= nIterations:
            V = np.max(self.mdp.R + self.mdp.discount * np.matmul(self.mdp.T,V_old), axis=0)
            epsilon = np.linalg.norm(V_old - V)
            if epsilon < tolerance:
                break
            else:
                V_old = V
                iterId += 1
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        policy = np.argmax(self.mdp.R + self.mdp.discount * np.matmul(self.mdp.T,V), axis=0)

        return np.array(policy)

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''
        

        R_pi = self.mdp.R[policy.tolist(), range(self.mdp.nStates)]
        T_pi = self.mdp.T[policy.tolist(), range(self.mdp.nStates)]

        V = np.linalg.solve(np.eye(self.mdp.nStates) - self.mdp.discount * T_pi, R_pi)

        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        policy_old = initialPolicy

        iterId = 1
        while iterId <= nIterations:
            V = self.evaluatePolicy(policy_old)
            policy = self.extractPolicy(V)
            if np.equal(policy, policy_old).all():
                break
            else:
                policy_old = policy
                iterId += 1

        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V_old = initialV
        R_pi = self.mdp.R[policy.tolist(), range(self.mdp.nStates)]
        T_pi = self.mdp.T[policy.tolist(), range(self.mdp.nStates)]
        
        iterId = 1
        while iterId <= nIterations:
            V = R_pi + self.mdp.discount * np.matmul(T_pi,V_old)
            epsilon = np.linalg.norm(V_old - V)
            if epsilon < tolerance:
                break
            else:
                V_old = V
                iterId += 1

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        V_old = initialV
        V = initialV
        policy = initialPolicy
        
        iterId = 1
        while iterId <= nIterations:
            [V, iterId_temp, epsilon] = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            policy = self.extractPolicy(V)
            epsilon = np.linalg.norm(V_old - V)
            if epsilon < tolerance:
                break
            else:
                V_old = V
                iterId += 1

        return [policy,V,iterId,epsilon]