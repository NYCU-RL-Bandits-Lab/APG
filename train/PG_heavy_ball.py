'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-08-04 09:14:40
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-08-08 20:44:14
FilePath: /mru/APG/train/PG_heavy_ball.py
Description: 

'''
# package
import os
import copy                               # important copy by value, not reference!!!...
from tqdm import tqdm
import numpy as np

# other .py
from Bellman import Bellman
from Saver import Saver


# -------------- Accelerated Policy Gradient --------------
class PG_heavy_ball_model(Bellman, Saver):

    def __init__(self, args: object, logger: object):

        # initialize class Bellman & Save
        Bellman.__init__(self, args)
        Saver.__init__(self, args, logger, "PG_heavy_ball")

        # logger to record
        self.logger = logger

        # how many seed num to run
        self.seed_num = args.seed_num

        # Storage (write into .parquet per (chunk_size) epoch)
        self.chunk_size = args.chunk_size

        # save path for .parquet
        self.save_path = os.path.join(self.logger.log_dir, 'PG_heavy_ball', f'mean.parquet')          
        
    
    # -------------- training loop --------------
    def learn(self, epoch: int):

        # log
        self.logger(f"Running PG Heavy Ball...", title=False)
                    
        for seed in range(self.seed_num):

            # set seed & saving path if stochastic
            self.set_seed_save_path(seed, self.logger)

            # set tqdm
            pbar = tqdm(range(epoch), position=3)

            # init theta
            theta_t = copy.deepcopy(self.theta_0)
            theta_t_1 = copy.deepcopy(self.theta_0)    # last theta (theta_{t-1})
            omega_t = copy.deepcopy(self.theta_0)
            mom_t = np.zeros_like(theta_t)
            grad_t = np.zeros_like(theta_t)
            epoch_record = np.zeros(shape=(self.chunk_size, len(self.record_columns)), dtype=np.float64)

            # run
            for timestep in pbar:

                # set pbar
                pbar.set_description(f"[PG Heavy Ball] Epoch {timestep:^10d}/{epoch:^10d}")

                # compute policy (pi)
                pi_t = self.compute_pi(theta_t)
                omega_pi_t = self.compute_pi(omega_t)

                # compute V, Q, Adv
                V_t, Q_t, Adv_t = self.compute_V_Q_Adv(pi_t)
                V_omega_t, Q_omega_t, Adv_omega_t = self.compute_V_Q_Adv(omega_pi_t)

                # compute discounted state visitation distribution
                d_t, d_rho_t = self.compute_d(pi_t)
                d_omega_t, d_rho_omega_t = self.compute_d(pi_t)

                # record
                epoch_record[timestep % self.chunk_size, :] = np.concatenate((pi_t, omega_pi_t, theta_t, omega_t, mom_t, grad_t, V_t, V_omega_t, Q_t, Adv_t, d_t, d_rho_t), axis=None)

                # policy gradient
                if self.stochastic:
                    pass
                else:
                    # true gradient update
                    grad_t = self.compute_true_grad(pi_t, Adv_t, d_rho_t)
                    theta_t =  copy.deepcopy(omega_t) + self.eta * grad_t
                    
                    # momentum update
                    mom_t = copy.deepcopy(theta_t - theta_t_1)
                    mom_t[~np.isfinite(mom_t)] = 0.
                    omega_t = copy.deepcopy(theta_t) + (0.5) * mom_t

                    # store theta_{t-1} for Polyak's accelerating
                    theta_t_1 = copy.deepcopy(theta_t)

                # set pbar
                pbar.set_postfix_str(f"V_t: {[round(v[0],3) for v in V_t]}")


                # save training process
                if ((timestep+1) % self.chunk_size == 0) or timestep == (epoch-1):
                    
                    # save to parquet
                    self.save(epoch_record)

                    # clear the array
                    epoch_record = np.zeros(shape=(self.chunk_size, len(self.record_columns)), dtype=np.float64)
        
        self.logger(f"Finish running PG Heavy Ball...", title=False)