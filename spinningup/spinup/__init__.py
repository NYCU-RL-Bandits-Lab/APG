'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-11-11 22:03:29
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-11-11 23:07:52
FilePath: /mru/Spinningup/spinningup/spinup/__init__.py
Description: 

'''
# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Algorithms
from spinup.algos.tf1.ddpg.ddpg import ddpg as ddpg_tf1
from spinup.algos.tf1.ppo.ppo import ppo as ppo_tf1
from spinup.algos.tf1.sac.sac import sac as sac_tf1
from spinup.algos.tf1.td3.td3 import td3 as td3_tf1
from spinup.algos.tf1.trpo.trpo import trpo as trpo_tf1
from spinup.algos.tf1.vpg.vpg import vpg as vpg_tf1

from spinup.algos.pytorch.ddpg.ddpg import ddpg as ddpg_pytorch
from spinup.algos.pytorch.ppo.ppo import ppo as ppo_pytorch
from spinup.algos.pytorch.sac.sac import sac as sac_pytorch
from spinup.algos.pytorch.td3.td3 import td3 as td3_pytorch
from spinup.algos.pytorch.trpo.trpo import trpo as trpo_pytorch
from spinup.algos.pytorch.vpg.vpg import vpg as vpg_pytorch
from spinup.algos.pytorch.vpg_sgd.vpg import vpg as vpg_sgd_pytorch
from spinup.algos.pytorch.vpg_sgd_nesterov.vpg import vpg as vpg_sgd_nesterov_pytorch
from spinup.algos.pytorch.vpg_sgd_hb.vpg import vpg as vpg_sgd_hb_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__