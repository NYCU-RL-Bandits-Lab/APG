# package
import os
import argparse

# path
import sys
sys.path.insert(1, './helper/')
sys.path.insert(1, './train/')

# logger
from loguru import logger
logger.remove()
my_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name:16.16}</cyan> | <cyan>{function:10.10}</cyan> | line:<cyan>{line:4d}</cyan> - <level>{message}</level>"
logger.add(sys.stderr, format=my_format, level='INFO')

# other .py
from train import *
from helper import *


def main():

    # -------------- Parameter --------------
    args = parse_args()


    # -------------- Logger --------------
    log_dir = make_dir(args)
    logger.add(os.path.join(log_dir, 'run.log'), format=my_format, level='DEBUG')
    logger.info(f'Log directory: {log_dir}')
    args.log_dir = log_dir


    # -------------- load the MDP environment from .yaml --------------
    args = load_MDP(args)


    # -------------- debug (lr, check initial_state_distribution_dict, reward_dict, initial_theta_dict, transition_prob_dict) --------------
    args = check_env(args)


    # -------------- Policy Iteration (PI) to find the optimal policy --------------
    PI = PI_model(args)
    optimal_policy, V_opts, V_opt, d_rho_opt = PI.learn(args.max_iter)
    opt = vars(args)
    opt.update({"optimal_policy": optimal_policy.tolist(), "V_opts": V_opts.squeeze().tolist(), "V_opt": V_opt.item(), "d_rho_opt": d_rho_opt.squeeze().tolist()})  # store optimal policy into args
    args = argparse.Namespace(**opt)

    logger.debug(f'='*100)
    logger.debug('Argparse:')
    for arg, value in vars(args).items():
        logger.debug(f'{arg} = {value}')
    logger.debug(f'='*100)
    
    # -------------- save param --------------
    save_param(args.__dict__, log_dir)


    # -------------- Run algo under true critic & softmax parameterization --------------
    # construct model
    models = {
        'PG': PG_model(args),
        'APG': APG_model(args),
        'NAPG': NAPG_model(args),
        'PG_adam': PG_adam_model(args),
        'PG_heavy_ball': PG_heavy_ball_model(args),
        'APG_adaptive': APG_adaptive_model(args),
        'NPG': NPG_model(args),
    }

    model = models[args.algo]
    model.learn(args.epoch)


    # -------------- plot --------------
    # construct plotter
    plotter = Plotter(args, log_dir)

    # data preprocess
    if args.stochastic:
        date_preprocessing(args.algo, args.seed_num, os.path.join(log_dir, args.algo))

    # plot under different size
    for size in args.graphing_size:
        
        plotter.plot_summary(size)


    # -------------- Training finished --------------
    open(os.path.join(log_dir, 'done'), 'a').close()
    logger.info(f'DONE')
    logger.info('')
    logger.info('#'*100)
    logger.info('#'*100)
    logger.info('#'*100)
    logger.info('')


if __name__ == '__main__':
    main()