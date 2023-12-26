'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-02-11 00:09:40
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-11-18 01:36:36
FilePath: /mru/APG/graph.py
Description: 
    
'''
# package
import os
import yaml
import argparse

# path
import sys
sys.path.insert(1, './helper/')

# other .py
from utils import Logger
from plot import Plotter


# -------------- Parameter --------------
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='./logs/test', help='root directory for log')
parser.add_argument('--algo', default='APG', help='algorithm to plot')
parser.add_argument('--graphing_size', default=[50], type=int, nargs='+', help='')
parser.add_argument('--plot_Summary', default=False, action='store_true')
parser.add_argument('--plot_Value', default=False, action='store_true')
parser.add_argument('--plot_LogLog', default=False, action='store_true')
parser.add_argument('--plot_MomGrad', default=False, action='store_true')
parser.add_argument('--plot_OneStep', default=False, action='store_true')
parser.add_argument('--plot_Q', default=False, action='store_true')
parser.add_argument('--plot_Pi', default=False, action='store_true')
parser.add_argument('--plot_Theta', default=False, action='store_true')
parser.add_argument('--plot_Adv', default=False, action='store_true')
parser.add_argument('--plot_adam', default=False, action='store_true')
parser.add_argument('--plot_Value_tmp', default=False, action='store_true')
args = parser.parse_args()

# load the arg from .yaml
opt = yaml.load(open(os.path.join(args.log_dir, "args.yaml")), Loader=yaml.FullLoader)
opt.update(vars(args))
args = argparse.Namespace(**opt)


# -------------- plot --------------
# Logger
logger = Logger(args)
logger(f"Start Plotting more figure", title=True)

# construct plotter
plotter = Plotter(args, logger)

# plot under different size
if args.plot_Summary:
    
    for size in args.graphing_size:
        
        plotter.plot_Summary(size, args.algo)

if args.plot_Value:
    
    for size in args.graphing_size:
        
        plotter.plot_Value(size, args.algo)

if args.plot_LogLog:
    
    for size in args.graphing_size:
        
        plotter.plot_LogLog(size)

if args.plot_MomGrad:
    
    for size in args.graphing_size:
        
        plotter.plot_MomGrad(size, args.algo)

if args.plot_OneStep:
    
    for size in args.graphing_size:
        
        plotter.plot_OneStep(size, args.algo)

if args.plot_Q:
    
    for size in args.graphing_size:
        
        plotter.plot_Q(size, args.algo)

if args.plot_Pi:
    
    for size in args.graphing_size:

        plotter.plot_Pi(args.algo)

if args.plot_Theta:
    
    for size in args.graphing_size:

        plotter.plot_Theta(args.algo)

if args.plot_Adv:
    
    for size in args.graphing_size:

        plotter.plot_Adv(args.algo)

if args.plot_adam:
    
    for size in args.graphing_size:

        plotter.plot_adam(args.algo)

if args.plot_Value_tmp:
    
    for size in args.graphing_size:

        plotter.plot_Value_tmp(size)

logger(f"Finish Plotting", title=True)