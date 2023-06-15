# package
import os
import time
import yaml
import shutil
import random
import argparse
import numpy as np
from datetime import datetime
from termcolor import colored, cprint


# -------------- load the MDP environment from .yaml --------------
def load_MDP(args, logger):

    # generate MDP
    if args.random_mdp:
        args.env = "Random MDP"
        logger("Randomly generating a MDP", title=True)
        assert args.state_action_num != [None, None], "input the state, action number so that I can generate MDP for you."
        mdp_info = generate_MDP(args.state_action_num[0], args.state_action_num[1])
        opt = vars(args)
        opt.update(mdp_info)
    
    # or load MDP
    else:
        assert os.path.exists(args.env), "No environment path in your parameters. Check the path and parameters.py"
        opt = vars(args)
        args = yaml.load(open(args.env), Loader=yaml.FullLoader)
        opt.update(args)
    args = argparse.Namespace(**opt)
    
    return args


# -------------- save parameters --------------
def save_param(data: dict, log_dir: str):

    with open(os.path.join(log_dir, "args.yaml"), 'w') as f:
        yaml.dump(data, f, Dumper=yaml.CDumper)


# -------------- Logger --------------
class Logger:

    def __init__(self, args):

        # reload logger
        if "log_dir" in vars(args).keys():
            # load log_dir
            self.log_dir = args.log_dir
        
        # construct logger
        else:
            # named file name as datetime if no specify
            if args.fname:
                self.log_dir = os.path.join(args.log_root, args.fname)
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(args.log_root, timestamp)
            
            # make directory
            self.make_directory(self.log_dir, args.fname, args.run_algos)

    def __call__(self, text: str, title: bool):

        # stdout
        if title:
            cprint(f"\n=========================== {text} ===========================", "green")
        else:
            cprint(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}", "blue")

        # file
        if title:
            with open('{}/train_record.txt'.format(self.log_dir), 'a') as train_record:
                train_record.write(f'\n=========================== {text} ===========================\n')
        else:
            with open('{}/train_record.txt'.format(self.log_dir), 'a') as train_record:
                train_record.write(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}\n")
    
    def make_directory(self, log_dir: str, fname: str, algos: list):

        # make directory
        if fname in ["testing", "test"] and os.path.isdir(log_dir):
            self("Removing the original test / testing file", title=True)
            shutil.rmtree(log_dir)
            os.mkdir(log_dir)
        
        elif os.path.isdir(log_dir):
            self("Directory already exist, change a file name", title=True)
            os._exit(1)
        
        else:
            path = str()
            for i in log_dir.split("/"):
                path = os.path.join(path, i)
                if path != "." and (not os.path.isdir(path)):
                    os.mkdir(path)     
        
        # make directory for specific algorithm
        for algo in algos:
            if not os.path.isdir(os.path.join(log_dir, algo)):
                os.mkdir(os.path.join(log_dir, algo))



# -------------- Debug --------------
def check_env(args, logger):

    # stochastic / not
    if not args.stochastic and args.seed_num != 1:
        args.seed_num = 1
        logger("True gradient has no randomness, automatically modifying seed num to be 1", title=True)

    # graphing size:
    if args.APG_graphing_size == [None]:
        args.APG_graphing_size = [args.APG_epoch_size]
        logger(f"Setting APG_graphing_size = APG_epoch_size = {args.APG_epoch_size}", title=True)
    if args.PG_graphing_size == [None]:
        args.PG_graphing_size = [args.PG_epoch_size]
        logger(f"Setting PG_graphing_size = PG_epoch_size = {args.APG_epoch_size}", title=True)

    # adapt lr to [Mei et al., ICML 2020]
    if args.gamma != 0:
        args.eta = pow(1-args.gamma, 3) / 8.0
        logger(f"Automatically set lr = (1-gamma)^3/8 = {args.eta}", title=True)

    # check state_num action_num
    assert max(args.state_num, args.action_num) <= 6,\
        "max(state_num, action_num)= {} might also work, but you have to modified the color list in the utils.plot_* funtion".format(max(args.state_num, args.action_num))

    # check initial_state_distribution_dict
    assert len(args.initial_state_distribution_dict) == args.state_num,\
        "len(args.initial_state_distribution_dict) = {} != {} = state_num".format(len(args.initial_state_distribution_dict), args.state_num)
    assert sum(args.initial_state_distribution_dict.values()) == 1,\
        "sum(args.initial_state_distribution_dict.items) = {} != 1".format(sum(args.initial_state_distribution_dict.items))

    # check reward_dict
    assert len(args.reward_dict) == args.state_num * args.action_num,\
        "len(args.reward_dict) = {} != {} = state_num*action_num".format(len(args.reward_dict), args.state_num * args.action_num)

    # check initial_theta_dict
    assert len(args.initial_theta_dict) == args.state_num,\
        "len(args.initial_theta_dict) = {} != {} = state_num".format(len(args.initial_theta_dict), args.state_num)
    for state in [f's{num}' for num in range(1, args.state_num+1)]:
        assert len(args.initial_theta_dict[state]) == args.action_num,\
            "len(args.initial_theta_dict[state]) = {} != {} = action".format(len(args.initial_theta_dict[state]), args.action_num)

    # check transition_prob_dict
    assert len(args.transition_prob_dict) == args.state_num * args.state_num * args.action_num,\
        "len(args.transition_prob_dict) = {} != {} = state_num*state_num*action_num".format(len(args.transition_prob_dict), args.state_num * args.state_num * args.action_num)
    for action in [f'a{num}' for num in range(1, args.action_num+1)]:
        for state in [f's{num}' for num in range(1, args.state_num+1)]:
            prob_sum = 0
            for terminal in [f's{num}' for num in range(1, args.state_num+1)]:
                prob_sum += args.transition_prob_dict[f'{state}{action}_{terminal}']
            assert round(prob_sum,4) == 1.0, "sum(P({}a[]_{})) need to be 1. But get {:.4f} here".format(state, terminal, prob_sum)
    
    return args


# -------------- data preprocessing --------------
def date_preprocessing(seed_num: int, log_dir: str):
    pass

    # # logger(f"Data Preprocessing...", title=True)

    # if seed_num > 1:
        
    #     start = time.time()
    #     print("\nPreprocessing...\n")

    #     # read parquet
    #     seed_df = pd.read_parquet(os.path.join(log_dir, f'seed_{0}.parquet'))

    #     # storing the same item based on different seed
    #     df = {}
    #     for item in df.keys():
    #         df[item] =  pd.DataFrame()
        
    #     # read all csv under different seed
    #     for seed in range(seed_num):
    #         seed_df = pd.read_parquet(os.path.join(log_dir, f'seed_{seed}.parquet'))
    #         for item in seed_df.keys():
    #             if seed == 0:
    #                 df[item] = seed_df[item]
    #             else:
    #                 df[item] = pd.concat([df[item], seed_df[item]], axis=1)

    #     # compute mean & std, save into .parquet
    #     mean_df = pd.DataFrame()
    #     std_df = pd.DataFrame()
    #     for item in df.keys():
    #         if ("pi" in item) or ("V" in item) or ("d_mu" in item):
    #             mean_df[item] = df[item].mean(axis=1).rename(item+'_mean')
    #             std_df[item] = df[item].std(axis=1).rename(item+'_std')
    #         else:
    #             mean_df[item] = df[item].mean(axis=1).rename(item+'_mean')
        
    #     # delete the memory monster
    #     del df

    #     # save file
    #     mean_df.to_parquet(os.path.join(log_dir, f'mean.parquet'))
    #     std_df.to_parquet(os.path.join(log_dir, f'std.parquet'))

    #     # delete the seed file
    #     for seed in range(seed_num):
    #         os.remove(os.path.join(log_dir, f'seed_{seed}.parquet'))
        
    #     print(f"It took {round(time.time()-start, 2)} sec for data preprocessing.\n")


# -------------- randomly genereate an MDP --------------
def generate_MDP(state_num: int, action_num: int):
    
    def softmax(x):
    
        f_x = np.exp(x) / np.sum(np.exp(x))
        return [float(x) for x in f_x]

    def prob_gen(num):

        prob = softmax([random.randint(0, 3) for _ in range(num)])
        prob = [round(x, 2) for x in prob]
        while sum(prob) != 1:
            gap = 1 - sum(prob)
            prob[random.randint(0, len(prob)-1)] += gap
            prob = [round(x, 2) for x in prob]
            if sum(prob) != 1:
                prob = softmax([random.randint(0, 3) for _ in range(num)])
                prob = [round(x, 2) for x in prob]
        
        return prob

    # initialize
    MDP = dict()
    
    # |S| & |A|
    MDP["state_num"] = state_num
    MDP["action_num"] = action_num

    # generate probability list
    pick_list = prob_gen(state_num)

    # initial state distribution
    initial_state_distribution_dict = dict()
    for (num, prob) in enumerate(pick_list):
        initial_state_distribution_dict[f"s{num+1}"] = prob
    MDP["initial_state_distribution_dict"] = initial_state_distribution_dict

    # initial theta dict
    initial_theta_dict = dict()
    for num in range(state_num):
        initial_theta_dict[f"s{num+1}"] = [random.randint(0, 5) for i in range(action_num)]
    MDP["initial_theta_dict"] = initial_theta_dict

    # reward dict
    reward_dict = dict()
    for s_num in range(state_num):
        record_list = []
        for a_num in range(action_num):
            random_reward = round(random.random(), 2)
            # avoid duplicate reward
            while round(random.random(), 2) in record_list:
                random_reward = round(random.random(), 2)
            record_list.append(round(random.random(), 2))
            reward_dict[f"s{s_num+1}_a{a_num+1}"] = round(random.random(), 2)
    MDP["reward_dict"] = reward_dict
    
    # dynamic dict
    transition_prob_dict = dict()
    for s_num in range(state_num):
        for a_num in range(action_num):
            # pick a transition prob to s' (terminal state)
            pick_list = prob_gen(state_num)
            for (terminal_num, prob) in enumerate(pick_list):
                transition_prob_dict[f"s{s_num+1}a{a_num+1}_s{terminal_num+1}"] = prob
    MDP["transition_prob_dict"] = transition_prob_dict

    return MDP