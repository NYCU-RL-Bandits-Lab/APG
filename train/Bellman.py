'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-06-14 22:43:04
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-10-26 11:43:47
FilePath: /mru/APG/train/Bellman.py
Description: 

'''
import copy   
import random
import numpy as np
from numpy.linalg import inv              # for matrix inverse


class Bellman:

    def __init__(self, args: object):

        # (s,a) number
        self.state_num = args.state_num
        self.action_num = args.action_num

        # sotchastic
        self.stochastic = args.stochastic

        # lr (depends on gamma, ref [Mei])
        self.eta = args.eta
        
        # discounted factor
        self.gamma = args.gamma

        # for transition and learn
        self.initial_state_distribution = np.zeros(shape=(self.state_num), dtype=np.float64)
        self.transition_prob_1 = np.zeros(shape=(self.state_num, self.state_num, self.action_num), dtype=np.float64)
        self.transition_prob_2 = np.zeros(shape=(self.state_num, self.action_num, self.state_num), dtype=np.float64)
        self.theta_0 = np.zeros(shape=(self.state_num, self.action_num), dtype=np.float64)
        self.delta_theta = np.zeros_like(self.theta_0, dtype=np.float64)
        self.reward = np.zeros(shape=(self.state_num, self.action_num), dtype=np.float64)
        
        for s_num in range(self.state_num):
            
            # initialize theta_0
            self.theta_0[s_num, :] = [i if i != "-inf" else -np.inf for i in args.initial_theta_dict[f"s{s_num+1}"]]

            # initialize rho
            self.initial_state_distribution[s_num] = args.initial_state_distribution_dict[f"s{s_num+1}"]

            for s2_num in range(self.state_num):
                
                for a_num in range(self.action_num):

                    # initialize transition probability
                    self.transition_prob_1[s2_num, s_num, a_num] = args.transition_prob_dict[f"s{s_num+1}a{a_num+1}_s{s2_num+1}"]
                    self.transition_prob_2[s_num, a_num, s2_num] = args.transition_prob_dict[f"s{s_num+1}a{a_num+1}_s{s2_num+1}"]

                    # initialize reward
                    self.reward[s_num, a_num] = args.reward_dict[f"s{s_num+1}_a{a_num+1}"]
            
            
    # -------------- for clipping the theta list (avoid math range error) --------------
    def clip(self, theta_t):
        
        max_theta = theta_t[np.isfinite(theta_t)].max() # max(theta_t)
        return theta_t - max_theta


    # -------------- compute policy weight under softmax parameterization --------------
    def compute_pi(self, theta_t: np.ndarray):
        
        # init
        pi_t = np.zeros_like(theta_t, dtype=np.float64)

        # softmax parameterization
        for num, theta in enumerate(theta_t):
            clip_theta = self.clip(copy.deepcopy(theta))
            clip_exp_theta = np.exp(clip_theta)
            denominator = sum(clip_exp_theta)
            pi_t[num, :] = clip_exp_theta / denominator

        return pi_t


    # -------------- compute Value function, Q function and Advantage function --------------
    def compute_V_Q_Adv(self, pi_t):

        # V = R_pi + gamma * P_pi * V    =>    V = (1 + gamma * P_pi)^{-1} R_pi
        R_pi = np.empty(shape=(self.state_num, 1), dtype=np.float64)
        P_pi = np.empty(shape=(self.state_num, self.state_num), dtype=np.float64)
        for s_num in range(self.state_num):
            R_pi[s_num, 0] = sum(pi_t[s_num, :] * self.reward[s_num, :])
            P_pi[:, s_num] = np.sum(pi_t * self.transition_prob_1[s_num], axis=1)
        V_t = np.matmul(inv(np.identity(self.state_num) - self.gamma * P_pi), R_pi)

        # Q = R + gamma * P * V
        Q_t = copy.deepcopy(self.reward)
        for s_num in range(self.state_num):
            Q_t[s_num, :] += self.gamma * np.squeeze(np.matmul(self.transition_prob_2[s_num], V_t))
        
        # Adv = Q - V
        Adv_t = Q_t - np.tile(V_t, (1, self.action_num))

        return V_t, Q_t, Adv_t


    # -------------- compute discounted state visitation distribution --------------
    def compute_d(self, pi_t):

        # compute P_pi in matrix form
        P_pi = np.empty(shape=(self.state_num, self.state_num), dtype=np.float64)
        for s_num in range(self.state_num):
            P_pi[:, s_num] = np.sum(pi_t * self.transition_prob_1[s_num], axis=1)

        # compute discounted state visitation distribution (d below) by solving matrix form
        d_t = (1 - self.gamma)*inv((np.identity(self.state_num) - self.gamma * P_pi))

        # weighted by rho
        d_rho_t = np.empty(shape=(self.state_num, 1), dtype=np.float64)
        for s_num in range(self.state_num):
            d_rho_t[s_num, 0] = np.inner(d_t[:, s_num], self.initial_state_distribution)
        
        return d_t, d_rho_t

        
    # -------------- compute stochastic grad --------------
    def compute_stochastic_grad(self, pi_t, Adv_t, d_mu_t):
        
        # print('='*100)
        # print('pi', pi_t)
        # print('='*100)
        # print('Adv', Adv_t)
        # print('='*100)
        # print('d_mu_t', d_mu_t)
        # print('='*100)

        # initialize
        delta_theta_t = np.zeros_like(pi_t, dtype=np.float64)

        # sample updated state (stochastic)
        s_t = 0 if self.state_num == 1 else random.choices(range(0, self.state_num), k=1, weights=np.squeeze(d_mu_t).tolist())[0]
        
        # sample updated action (stochastic)
        a_t = random.choices(range(0,len(pi_t[s_t])), k=1, weights=pi_t[s_t])[0]
        
        # print('s_t', s_t)
        # print('='*100)
        # print('a_t', a_t)
        # print('='*100)
        

        # stochastic PG update (batch size = 1)
        delta_theta_t[s_t] = [-pi * Adv_t[s_t, a_t] for pi in pi_t[s_t]]
        delta_theta_t[s_t, a_t] += Adv_t[s_t, a_t]
        delta_theta_t *= (1.0 / (1.0-self.gamma))

        # print('delta_theta_t', delta_theta_t)
        # print('='*100)
        # import os
        # os._exit(1)

        return delta_theta_t, (s_t+1, a_t+1)
        

    # -------------- compute true grad --------------
    def compute_true_grad(self, pi_t, Adv_t, d_mu_t):

        # grad = [ 1 / (1 - gamma) ] * pi * Adv * d  (ref to Agarwal work)
        delta_theta_t = np.zeros_like(pi_t, dtype=np.float64)
        delta_theta_t = pi_t * Adv_t * np.tile(d_mu_t, (1, self.action_num)) * (1.0 / (1.0 - self.gamma))

        return delta_theta_t