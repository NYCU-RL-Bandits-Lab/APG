# package
import os
import copy                               # important copy by value, not reference!!!...
from tqdm import tqdm
import numpy as np

# other .py
from Bellman import Bellman
from Saver import Saver

# -------------- Policy Gradient --------------
class PG_model(Bellman, Saver):

    def __init__(self, args: object, logger: object):
        
        # initialize class Bellman & Save
        Bellman.__init__(self, args)
        Saver.__init__(self, args, logger, "PG")
        
        # logger to record
        self.logger = logger

        # how many seed num to run
        self.seed_num = args.seed_num

        # Storage (write into .parquet per (chunk_size) epoch)
        self.chunk_size = args.chunk_size

        # save path for .parquet
        self.save_path = os.path.join(self.logger.log_dir, 'PG', f'mean.parquet')


    # -------------- training loop --------------
    def learn(self, epoch: int):
        
        # log
        self.logger(f"Running PG...", title=False)

        for seed in range(self.seed_num):

            # set seed & saving path if stochastic
            self.set_seed_save_path(seed, self.logger)

            # set tqdm
            pbar = tqdm(range(epoch), position=0)

            # init theta
            theta_t = copy.deepcopy(self.theta_0)
            delta_theta_t = np.zeros_like(theta_t)
            epoch_record = np.zeros(shape=(self.chunk_size, len(self.record_columns)), dtype=np.float64)
            selection = [1, 1]      # [selected state, selected action]

            # run
            for timestep in pbar:

                # set pbar
                pbar.set_description(f"[PG]  Epoch {timestep:^10d}/{epoch:^10d}")

                # compute policy (pi)
                pi_t = self.compute_pi(theta_t)

                # compute V, Q, Adv
                V_t, Q_t, Adv_t = self.compute_V_Q_Adv(pi_t)

                # compute discounted state visitation distribution
                d_t, d_rho_t = self.compute_d(pi_t)

                # record
                record = np.concatenate((pi_t, theta_t, delta_theta_t, V_t, Q_t, Adv_t, d_t, d_rho_t), axis=None)
                record = np.concatenate((record, selection)) if self.seed_num > 1 else record
                epoch_record[timestep % self.chunk_size, :] = record

                # policy gradient
                if self.stochastic:
                    delta_theta_t, selection = self.compute_stochastic_grad(pi_t, Adv_t, d_rho_t)
                else:
                    delta_theta_t = self.compute_true_grad(pi_t, Adv_t, d_rho_t)

                # one-step update
                theta_t += self.eta * delta_theta_t

                # set pbar
                pbar.set_postfix_str(f"V_t: {[round(v[0],3) for v in V_t]}")

                # save training process
                if ((timestep+1) % self.chunk_size == 0) or timestep == (epoch-1):
                    
                    # save to parquet
                    self.save(epoch_record)

                    # clear the array
                    epoch_record = np.zeros(shape=(self.chunk_size, len(self.record_columns)), dtype=np.float64)
        
        self.logger(f"Finish running PG...", title=False)