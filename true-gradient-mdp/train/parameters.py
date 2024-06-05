# package
import argparse

# -------------- Parameter --------------
def parse_args():
    parser = argparse.ArgumentParser()

    # train
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size', default=10, type=int, help='frequency for saving data, set large if RAM is small')
    parser.add_argument('--eta', default=0.4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=9e-1, type=float, help='mdp discounted factor')
    parser.add_argument('--algo', default="APG", type=str, choices=['PG', 'APG', 'NAPG', 'PG_adam', 'PG_heavy_ball', 'APG_adaptive', 'NPG', 'APG_GNPG'], help='algorithm to be run')
    parser.add_argument('--seed_num', default=3, type=int, help='number of seed to be run while running stochastic algorithm')
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--epoch', default=2000, type=int, help='number of epoch to run')
    
    # MDP env
    parser.add_argument('--env', type=str, default="./mdp_env/mdp_5s5a_uniform.yaml")
    parser.add_argument('--state_action_num', default=[None, None], type=int, nargs='+', help='params for random generate environment')
    parser.add_argument('--random_mdp', default=False, action='store_true')

    # graphing
    parser.add_argument('--graphing_size', default=[None], type=int, nargs='+', help='')

    # policy iteration
    parser.add_argument('--max_iter', default=100000000, type=float, help='PI maximum iteration')

    # root
    parser.add_argument('--fname', default=None, help='log directory name')
    parser.add_argument('--log_root', default='./logs', help='root directory for log')
    
    args = parser.parse_args()
    return args