#!/bin/bash
./run_mdp_5s5a_uniform_init_apg.sh
./run_mdp_5s5a_hard_init_apg.sh
./run_mdp_5s5a_uniform_init_stochastic.sh
./run_mdp_5s5a_hard_init_stochastic.sh 
./run_bandit_non_monotone.sh
./run_mdp_5s5a_uniform_init.sh
./run_mdp_5s5a_hard_init.sh
./run_mdp_5s5a_uniform_init_adaptive_apg.sh
./run_mdp_5s5a_uniform_init_apg_gnpg.sh