# package
import numpy as np
from loguru import logger

# other .py
from Bellman import Bellman

# -------------- Policy Gradient --------------
class PI_model(Bellman):

    def __init__(self, args: object):

        # initialize class Bellman & Save
        Bellman.__init__(self, args)


    # -------------- training loop --------------
    def learn(self, epoch: int):

        # random initialize policy (pi)
        pi_t = np.random.randint(self.action_num, size=self.state_num)
        
        # to one hot
        pi_t = np.eye(self.action_num)[pi_t]


        # run
        for timestep in range(1, epoch + 1):

            # terminal flag
            terminal = True

            # compute V, Q
            V_t, Q_t, _, _ = self.compute_V_Q_Adv(pi_t)

            # PI
            for s_num in range(self.state_num):

                # find the best action
                best_action = Q_t[s_num].tolist().index(max(Q_t[s_num]))
                
                # detect whether terminate (V_{t+1} = V_{t})
                if pi_t[s_num][best_action] != 1:
                    terminal = False

                # policy iteration (set P(action with maximum Q) = 1)
                pi_t[s_num] = np.zeros(shape=(self.action_num))
                pi_t[s_num][best_action] = 1

            if terminal:
                logger.info(f"Policy Iteration terminal at epoch: {timestep}, V: {V_t.squeeze()}, optimal policy: {np.argmax(pi_t, axis=1)}")
                break
            
        V_opts, _, _, V_opt = self.compute_V_Q_Adv(pi_t)
        _, d_rho_opt = self.compute_d(pi_t)
        
        return np.argmax(pi_t, axis=1), V_opts, V_opt, d_rho_opt