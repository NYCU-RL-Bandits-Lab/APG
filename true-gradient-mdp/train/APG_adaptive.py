# package
import os
import copy                               # important copy by value, not reference!!!...
import numpy as np
from loguru import logger

# other .py
from Bellman import Bellman
from Saver import Saver


# -------------- Accelerated Policy Gradient --------------
class APG_adaptive_model(Bellman, Saver):

    def __init__(self, args: object):

        # initialize class Bellman & Save
        Bellman.__init__(self, args)
        Saver.__init__(self, args, "APG_adaptive")

        # how many seed num to run
        self.seed_num = args.seed_num

        # Storage (write into .parquet per (chunk_size) epoch)
        self.chunk_size = args.chunk_size

        # save path for .parquet
        self.save_path = os.path.join(args.log_dir, 'APG_adaptive', f'mean.parquet') 

        # V(optimum)
        self.V_opt = args.V_opt  

    
    # -------------- training loop --------------
    def learn(self, epoch: int):

        # log
        logger.info(f"{'[APG_adaptive]':16s} Start Running")
                    
        for seed in range(self.seed_num):

            # set seed & saving path if stochastic
            self.set_seed_save_path(seed)

            # init theta
            theta_t = copy.deepcopy(self.theta_0)
            theta_t_1 = copy.deepcopy(self.theta_0)    # last theta (theta_{t-1})
            omega_t = copy.deepcopy(self.theta_0)
            mom_t = np.zeros_like(theta_t)
            grad_t = np.zeros_like(theta_t)
            selection = [1, 1]      # [selected state, selected action]

            # run
            for timestep in range(epoch):

                # log info
                if ((timestep+1) % self.chunk_size == 0) or (timestep == 0):
                    logger.debug(f"[APG_adaptive] Epoch {timestep:^10d}/{epoch:^10d}")

                # compute policy (pi)
                pi_t = self.compute_pi(theta_t)
                omega_pi_t = self.compute_pi(omega_t)

                # compute V, Q, Adv
                V_t, Q_t, Adv_t, V_rho_t = self.compute_V_Q_Adv(pi_t)
                V_omega_t, Q_omega_t, Adv_omega_t, V_omega_rho_t = self.compute_V_Q_Adv(omega_pi_t)

                # compute discounted state visitation distribution
                d_t, d_rho_t = self.compute_d(pi_t)
                d_omega_t, d_rho_omega_t = self.compute_d(pi_t)

                # record
                if ((timestep+1) % self.chunk_size == 0) or (timestep == 0) or (timestep in self.save_time):
                    record = np.concatenate((
                        timestep, 
                        np.log(timestep+1),
                        pi_t, 
                        omega_pi_t, 
                        theta_t, 
                        omega_t, 
                        mom_t, 
                        grad_t, 
                        V_t, 
                        V_rho_t, 
                        -np.log(self.V_opt - V_rho_t.item() + 1e-100),
                        V_omega_t, 
                        V_omega_rho_t, 
                        -np.log(self.V_opt - V_omega_rho_t.item() + 1e-100),
                        Q_t, 
                        Adv_t, 
                        d_t, 
                        d_rho_t,
                    ), axis=None)
                    record = np.concatenate((record, selection)) if self.seed_num > 1 else record
                    self.save(record, (timestep==epoch-1))

                # policy gradient
                if self.stochastic:
                    grad_t, selection = self.compute_stochastic_grad(omega_pi_t, Adv_omega_t, d_rho_omega_t)
                else:
                    grad_t = self.compute_true_grad(omega_pi_t, Adv_omega_t, d_rho_omega_t)
                
                theta_t =  copy.deepcopy(omega_t) + (0.5) * (float(timestep + 1) / (timestep + 2)) * self.eta * grad_t * min(np.power(5., timestep+1), 1. / (np.linalg.norm(grad_t) + 1e-100))
                
                # momentum update
                mom_t = copy.deepcopy(theta_t - theta_t_1)
                mom_t[~np.isfinite(mom_t)] = 0.
                phi_t = copy.deepcopy(theta_t) + (float(timestep) / (timestep + 3)) * mom_t
                phi_pi_t = self.compute_pi(phi_t)
                V_phi_t, _, _, V_phi_rho_t = self.compute_V_Q_Adv(phi_pi_t)

                # monotone
                monotone = V_phi_rho_t.item() + 1e-12 >= V_rho_t.item()
                if monotone:
                    omega_t = copy.deepcopy(phi_t)
                else:
                    omega_t = copy.deepcopy(theta_t)
                    logger.debug(f"[APG_adaptive] {timestep}: non-monotone, V_phi_rho_t = {V_phi_rho_t.item()}, V_rho_t = {V_rho_t.item()}")

                # store theta_{t-1} for Nesterov's accelerating
                theta_t_1 = copy.deepcopy(theta_t)

        logger.info(f"[APG_adaptive] Finish Running")