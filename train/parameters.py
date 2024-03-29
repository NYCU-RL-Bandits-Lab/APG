'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-05-11 15:32:32
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-11-29 12:30:07
FilePath: /mru/APG/train/parameters.py
Description: 

'''
# package
import argparse

# -------------- Parameter --------------
def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', default=100000, type=int, help='Frequency for saving data, set small if RAM is small')
    parser.add_argument('--eta', default=0.4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=9e-1, type=float, help='discounted')
    parser.add_argument('--run_algos', default=["APG", "PG", "PG_heavy_ball", "PG_adam"], type=str, nargs='+', help='')
    parser.add_argument('--seed_num', default=3, type=int, help='seed number')
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--APG_epoch_size', default=1000, type=int, help='APG epoch_size')
    parser.add_argument('--APG_adaptive_epoch_size', default=1000, type=int, help='APG epoch_size')
    parser.add_argument('--PG_adam_epoch_size', default=1000, type=int, help='PG_adam epoch_size')
    parser.add_argument('--PG_heavy_ball_epoch_size', default=1000, type=int, help='PG_heavy_ball epoch_size')
    parser.add_argument('--PG_epoch_size', default=1000, type=int, help='PG epoch_size')
    
    # MDP env
    parser.add_argument('--env', type=str, default="./mdp_env/test.yaml")
    parser.add_argument('--state_action_num', default=[None, None], type=int, nargs='+', help='params for random generate environment')
    parser.add_argument('--random_mdp', default=False, action='store_true')

    # graphing
    parser.add_argument('--APG_graphing_size', default=[None], type=int, nargs='+', help='')
    parser.add_argument('--APG_adaptive_graphing_size', default=[None], type=int, nargs='+', help='')
    parser.add_argument('--PG_adam_graphing_size', default=[None], type=int, nargs='+', help='')
    parser.add_argument('--PG_heavy_ball_graphing_size', default=[None], type=int, nargs='+', help='')
    parser.add_argument('--PG_graphing_size', default=[None], type=int, nargs='+', help='')

    # policy iteration
    parser.add_argument('--max_iter', default=100000000, type=float, help='PI maximum iteration')

    # root
    parser.add_argument('--fname', default=None, help='log directory name')
    parser.add_argument('--log_root', default='./logs', help='root directory for log')
    
    args = parser.parse_args()
    return args