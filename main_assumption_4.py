'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-11-27 11:00:55
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-11-27 12:14:24
FilePath: /mru/APG/main_assumption_4.py
Description: 

'''
# package
import os
import json
import yaml
import argparse
import numpy as np
from itertools import product
import multiprocessing as mp

# path
import sys
sys.path.insert(1, './helper/')
sys.path.insert(1, './train/')

# other .py
from utils import generate_MDP, load_MDP, save_param, Logger, check_env, date_preprocessing
from plot import Plotter
from Bellman import Bellman


# -------------- Parameter --------------
def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', default=0.4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=9e-1, type=float, help='discounted')
    parser.add_argument('--run_algos', default=[], type=str, nargs='+', help='')
    parser.add_argument('--seed_num', default=3, type=int, help='seed number')
    parser.add_argument('--stochastic', default=False, action='store_true')
   
    # MDP env
    parser.add_argument('--env', type=str, default="./mdp_env/test.yaml")
    parser.add_argument('--state_action_num', default=[None, None], type=int, nargs='+', help='params for random generate environment')
    parser.add_argument('--random_mdp', default=False, action='store_true')

    # root
    parser.add_argument('--fname', default='test', help='log directory name')
    parser.add_argument('--log_root', default='./logs/assum-4', help='root directory for log')
    
    args = parser.parse_args()
    return args


def main():

    # -------------- Parameter --------------
    args = parse_args()


    # -------------- Logger --------------
    logger = Logger(args)
    log_dir = logger.log_dir
    

    # -------------- load the MDP environment from .yaml --------------
    args = load_MDP(args, logger)


    # -------------- save param --------------
    save_param(args.__dict__, log_dir)


    # -------------- computer value under deterministic policy --------------
    bellman = Bellman(args)
    deterministic_policy = list()
    value = list()
    tmp = dict()
    for numbers in product(range(args.state_action_num[1]), repeat=args.state_action_num[0]):
        policy = np.array(
            [
                numbers[i] for i in range(args.state_action_num[0])
            ]
        )
        onehot = np.zeros((args.state_action_num[0], args.state_action_num[1]))
        onehot[np.arange(len(policy)), policy] = 1
        v, _, _ = bellman.compute_V_Q_Adv(onehot)
        v_rho = np.sum(v.squeeze() * np.array(list(args.initial_state_distribution_dict.values())))
        deterministic_policy.append(
            onehot
        )
        value.append(v_rho)
        tmp[v_rho] = policy.tolist()
    
    with open(os.path.join(logger.log_dir, 'policy-value.json'), 'w') as fp:
        json.dump(tmp, fp, sort_keys=True, indent=4)
    

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(6,3))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_alpha(0.2)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('V(ρ)')
    ax.scatter(value, np.zeros(len(value)), marker='|', c='blue')

    plt.savefig(os.path.join(logger.log_dir, f'policy-value.png'))


if __name__ == '__main__':
    main()