'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-06-15 13:39:44
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-08-08 20:15:10
FilePath: /mru/APG/train/Saver.py
Description: 

'''
# package
import os
import numpy as np
import pandas as pd

class Saver:

    def __init__(self, args: object, log_dir: str, algo: str):

        # algo
        self.algo = algo

        # stochastic
        self.stochastic = args.stochastic

        # log_dir
        self.log_dir = log_dir

        # optimal policy reached by Policy Iteration
        self.optimal_policy = args.optimal_policy

        # state-action name including the optimal action a*
        self.state_action_pair = dict()
        for (s_num, state) in enumerate([f's{s_num+1}' for s_num in range(self.state_num)]):
            self.state_action_pair[state] = [f'a{a_num+1}' for a_num in range(self.action_num)]
            self.state_action_pair[state][self.optimal_policy[s_num]] = "a*"
        
        # recording columns for .parquet 
        if self.algo in ["PG", "PG_adam"]:
            self.record_columns = [f'{state}_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_delta_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"V({state})" for state in self.state_action_pair.keys()] \
                                + [f"Q({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"Adv({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"d_{state1}({state2})" for state1 in self.state_action_pair.keys() for state2 in self.state_action_pair.keys()] \
                                + [f"d_rho({state})" for state in self.state_action_pair.keys()]

        elif self.algo in ["APG", "PG_heavy_ball"]:
            self.record_columns = [f'{state}_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                +[f'{state}_omega_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_omega_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_mom_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f'{state}_grad_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"V({state})" for state in self.state_action_pair.keys()] \
                                + [f"V_omega({state})" for state in self.state_action_pair.keys()] \
                                + [f"Q({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"Adv({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                                + [f"d_{state1}({state2})" for state1 in self.state_action_pair.keys() for state2 in self.state_action_pair.keys()] \
                                + [f"d_rho({state})" for state in self.state_action_pair.keys()]

        
    # -------------- save --------------
    def save(self, epoch_record: np.ndarray):
        
        # clear empty row
        epoch_record = epoch_record[~np.all(epoch_record == 0, axis=1)]

        # to dataframe
        df = pd.DataFrame(epoch_record, columns = self.record_columns, dtype='float64')

        # save & create / append
        if not os.path.isfile(self.save_path):
            df.to_parquet(self.save_path, engine='fastparquet')
        else:
            df.to_parquet(self.save_path, engine='fastparquet', append=True)

    
    # -------------- set seed (for stochastic) --------------
    def set_seed_save_path(self, seed, logger):
        
        if self.stochastic:

            # set seed
            self.logger(f"Set seed: {seed}", title=True)
            np.random.seed(seed_num)

            # set save path
            self.logger(f"Set saving path: {os.path.join(self.logger.log_dir, self.algo, f'seed_{seed}.parquet')}", title=True)
            self.save_path = os.path.join(self.log_dir, self.algo, f'seed_{seed}.parquet')
        
        else:

            self.logger(f"True gradient, no seed", title=True)