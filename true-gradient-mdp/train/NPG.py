# package
import os
import copy                               # important copy by value, not reference!!!...
import numpy as np
from loguru import logger

# other .py
from Bellman import Bellman
from Saver import Saver

# -------------- Policy Gradient --------------
class NPG_model(Bellman, Saver):

    def __init__(self, args: object):
        
        # initialize class Bellman & Save
        Bellman.__init__(self, args)
        Saver.__init__(self, args, "NPG")
        
        # how many seed num to run
        self.seed_num = args.seed_num

        # Storage (write into .parquet per (chunk_size) epoch)
        self.chunk_size = args.chunk_size

        # save path for .parquet
        self.save_path = os.path.join(args.log_dir, 'NPG', f'mean.parquet')

        # V(optimum)
        self.V_opt = args.V_opt


    # -------------- training loop --------------
    def learn(self, epoch: int):
        
        # log
        logger.info(f"[NPG] Start Running")

        for seed in range(self.seed_num):

            # set seed & saving path if stochastic
            self.set_seed_save_path(seed)

            # init theta
            theta_t = copy.deepcopy(self.theta_0)
            delta_theta_t = np.zeros_like(theta_t)
            selection = [1, 1]      # [selected state, selected action]

            # run
            for timestep in range(epoch):

                # log info
                if ((timestep+1) % self.chunk_size == 0) or (timestep == 0):
                    logger.debug(f"[NPG] Epoch {timestep:^10d}/{epoch:^10d}")

                # compute policy (pi)
                pi_t = self.compute_pi(theta_t)

                # compute V, Q, Adv
                V_t, Q_t, Adv_t, V_rho_t = self.compute_V_Q_Adv(pi_t)

                # compute discounted state visitation distribution
                d_t, d_rho_t = self.compute_d(pi_t)

                # record
                if ((timestep+1) % self.chunk_size == 0) or (timestep == 0) or (timestep in self.save_time):
                    record = np.concatenate((
                        timestep, 
                        np.log(timestep+1),
                        pi_t, 
                        theta_t, 
                        delta_theta_t, 
                        V_t, 
                        V_rho_t, 
                        -np.log(self.V_opt - V_rho_t.item()),
                        Q_t, 
                        Adv_t, 
                        d_t, 
                        d_rho_t
                    ), axis=None)
                    record = np.concatenate((record, selection)) if self.seed_num > 1 else record
                    self.save(record, (timestep==epoch-1))

                # policy gradient
                if self.stochastic:
                    raise ValueError
                else:
                    delta_theta_t = self.compute_natural_true_grad(pi_t, Adv_t, d_rho_t)

                # one-step update
                theta_t += self.eta * delta_theta_t * 1e-3

        logger.info(f"[NPG] Finish Running")