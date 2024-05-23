# package
import os
import numpy as np
import pandas as pd
from loguru import logger

class Saver:

    def __init__(self, args: object, algo: str):

        # algo
        self.algo = algo

        # stochastic
        self.stochastic = args.stochastic

        # log_dir
        self.log_dir = args.log_dir

        # optimal policy reached by Policy Iteration
        self.optimal_policy = args.optimal_policy

        # state-action name including the optimal action a*
        self.state_action_pair = dict()
        for (s_num, state) in enumerate([f's{s_num+1}' for s_num in range(self.state_num)]):
            self.state_action_pair[state] = [f'a{a_num+1}' for a_num in range(self.action_num)]
            self.state_action_pair[state][self.optimal_policy[s_num]] = "a*"
        
        # recording columns for .parquet 
        if self.algo in ["PG", "NPG"]:
            self.record_columns = \
                ['timestep', 'log(timestep)'] \
                + [f'{state}_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_delta_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"V({state})" for state in self.state_action_pair.keys()] \
                + [f"V(rho)", "-log(V_opt-V(rho))"] \
                + [f"Q({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"Adv({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"d_{state1}({state2})" for state1 in self.state_action_pair.keys() for state2 in self.state_action_pair.keys()] \
                + [f"d_rho({state})" for state in self.state_action_pair.keys()]

        elif self.algo in ["APG", "NAPG", "PG_heavy_ball", "APG_adaptive"]:
            self.record_columns = \
                ['timestep', 'log(timestep)'] \
                + [f'{state}_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_omega_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_omega_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_mom_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_grad_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"V({state})" for state in self.state_action_pair.keys()] \
                + [f"V(rho)", "-log(V_opt-V(rho))"] \
                + [f"V_omega({state})" for state in self.state_action_pair.keys()] \
                + [f"V_omega(rho)", "-log(V_opt-V_omega(rho))"] \
                + [f"Q({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"Adv({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"d_{state1}({state2})" for state1 in self.state_action_pair.keys() for state2 in self.state_action_pair.keys()] \
                + [f"d_rho({state})" for state in self.state_action_pair.keys()]
        
        elif self.algo in ["PG_adam"]:
            self.record_columns = \
                ['timestep', 'log(timestep)'] \
                + [f'{state}_pi_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'{state}_delta_theta_{action}' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"V({state})" for state in self.state_action_pair.keys()] \
                + [f"V(rho)", "-log(V_opt-V(rho))"] \
                + [f"Q({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"Adv({state},{action})" for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f"d_{state1}({state2})" for state1 in self.state_action_pair.keys() for state2 in self.state_action_pair.keys()] \
                + [f"d_rho({state})" for state in self.state_action_pair.keys()] \
                + [f'm_t({state},{action})' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'v_t({state},{action})' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'm_t_hat({state},{action})' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]] \
                + [f'v_t_hat({state},{action})' for state in self.state_action_pair.keys() for action in self.state_action_pair[state]]

        if args.seed_num > 1:
            self.record_columns += ['stochastic_selection_state', 'stochastic_selection_action']
        
        # recorders
        self.records_counter = 0
        self.records_size = 100000
        self.records = np.zeros(shape=(self.records_size, len(self.record_columns)), dtype=np.float64)

        # saving time zone (to ensure log-log plot is correct)
        f1 = lambda x: int(np.exp(x)) * 1 // 4
        f2 = lambda x: int(np.exp(x)) * 2 // 4
        f3 = lambda x: int(np.exp(x)) * 3 // 4
        f4 = lambda x: int(np.exp(x))
        self.save_time = [f(x) for x in range(1, int(np.log(args.epoch)+1)) for f in (f1, f2, f3, f4)]
        
    # -------------- save --------------
    def save(self, record: np.ndarray, last_epoch: bool):
        
        self.records[self.records_counter, :] = record
        self.records_counter += 1

        # save parquet
        # if np.all(~np.all(self.records == 0, axis=1)) or last_epoch:
        if (self.records_counter == self.records_size) or last_epoch:
            
            # clear empty row
            self.records = self.records[~np.all(self.records == 0, axis=1)]

            # to dataframe
            df = pd.DataFrame(self.records, columns = self.record_columns, dtype='float64')

            # save & create / append
            if not os.path.isfile(self.save_path):
                df.to_parquet(self.save_path, engine='fastparquet')
            else:
                df.to_parquet(self.save_path, engine='fastparquet', append=True)
            
            # reset records & counter
            self.records_counter = 0
            self.records = np.zeros(shape=(self.records_size, len(self.record_columns)), dtype=np.float64)

    
    # -------------- set seed (for stochastic) --------------
    def set_seed_save_path(self, seed):
        
        if self.stochastic:

            # set seed
            logger.info(f"[{self.algo}] Set seed: {seed}")
            np.random.seed(seed)

            # set save path
            logger.info(f"[{self.algo}] Set saving path: {os.path.join(self.log_dir, self.algo, f'seed_{seed}.parquet')}")
            self.save_path = os.path.join(self.log_dir, self.algo, f'seed_{seed}.parquet')
        
        else:

            logger.info(f"[{self.algo}] True gradient, no seed")