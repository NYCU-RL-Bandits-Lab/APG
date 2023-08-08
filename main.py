'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-01-30 16:33:40
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-08-08 20:27:30
FilePath: /mru/APG/main.py
Description: 
    Test convergence of APG & PG Actor-Critic under softmax parameterization
    Work for bandit & simple MDP
    Only plot the summary, see plot.py for more graphing function
'''

# package
import os
import yaml
import argparse
import multiprocessing as mp

# path
import sys
sys.path.insert(1, './helper/')
sys.path.insert(1, './train/')

# other .py
from utils import generate_MDP, load_MDP, save_param, Logger, check_env, date_preprocessing
from plot import Plotter
from PI import PI_model
from PG import PG_model
from APG import APG_model
from PG_adam import PG_adam_model
from PG_heavy_ball import PG_heavy_ball_model
from parameters import parse_args


def main():

    # -------------- Parameter --------------
    args = parse_args()


    # -------------- Logger --------------
    logger = Logger(args)
    log_dir = logger.log_dir
    

    # -------------- load the MDP environment from .yaml --------------
    args = load_MDP(args, logger)


    # -------------- debug (lr, check initial_state_distribution_dict, reward_dict, initial_theta_dict, transition_prob_dict) --------------
    args = check_env(args, logger)


    # -------------- Policy Iteration (PI) to find the optimal policy --------------
    PI = PI_model(args, logger)
    optimal_policy, V_opts, V_opt, d_rho_opt = PI.learn(args.max_iter)
    opt = vars(args)
    opt.update({"optimal_policy": optimal_policy.tolist(), "V_opts": V_opts.squeeze().tolist(), "V_opt": V_opt.item(), "d_rho_opt": d_rho_opt.squeeze().tolist()})  # store optimal policy into args
    args = argparse.Namespace(**opt)


    # -------------- save param --------------
    save_param(args.__dict__, log_dir)


    # -------------- APG & PG under true credict & softmax parameterization --------------
    logger(f"Running", title=True)

    if "PG" in args.run_algos and "APG" in args.run_algos and "PG_heavy_ball" in args.run_algos:
        
        # construct pool
        pool = mp.Pool()
        
        # construct model
        PG = PG_model(args, logger)
        APG = APG_model(args, logger)
        PG_adam = PG_adam_model(args, logger)
        PG_heavy_ball = PG_heavy_ball_model(args, logger)

        # multiprocessing
        pool.apply_async(PG.learn, args = (args.PG_epoch_size, ))
        pool.apply_async(APG.learn, args = (args.APG_epoch_size, ))
        pool.apply_async(PG_adam.learn, args = (args.PG_adam_epoch_size, ))
        pool.apply_async(PG_heavy_ball.learn, args = (args.PG_heavy_ball_epoch_size, ))
        pool.close()
        pool.join()

    else:
        
        if "PG" in args.run_algos:

            # construct model
            PG = PG_model(args, logger)

            # run
            PG.learn(args.PG_epoch_size)
    
        if "APG" in args.run_algos:

            # construct model
            APG = APG_model(args, logger)

            # run
            APG.learn(args.APG_epoch_size)
        
        if "PG_heavy_ball" in args.run_algos:

            # construct model
            PG_heavy_ball = PG_heavy_ball_model(args, logger)

            # run
            PG_heavy_ball.learn(args.PG_heavy_ball_epoch_size)
        
        if "PG_adam" in args.run_algos:

            # construct model
            PG_adam = PG_adam_model(args, logger)

            # run
            PG_adam.learn(args.PG_adam_epoch_size)


    # -------------- plot --------------
    # construct plotter
    plotter = Plotter(args, logger)

    for algo in args.run_algos:
        
        # data preprocess
        if args.stochastic:
            date_preprocessing(args.seed_num, os.path.join(log_dir, algo))

        if algo == "PG":
            graphing_size = args.PG_graphing_size 
        elif algo == "APG":
            graphing_size = args.APG_graphing_size
        elif algo == "PG_adam":
            graphing_size = args.PG_adam_graphing_size
        elif algo == "PG_heavy_ball":
            graphing_size = args.PG_heavy_ball_graphing_size

        
        # plot under different size
        for size in graphing_size:
            
            logger(f"Plotting {algo}_train_stats_{size}.png", title=True)
            plotter.plot_Summary(size, algo)


    # -------------- Training finished --------------
    open(os.path.join(log_dir, 'done'), 'a').close()
    logger(f"Finished training", title=True)


if __name__ == '__main__':
    main()