# package
import os
import copy
import time
import yaml
import shutil
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')                      # backend
import matplotlib.pyplot as plt


# -------------- Plotting --------------
class Plotter:

    def __init__(self, args: object, logger: object):

        # logger to record
        self.logger = logger

        # epoch size
        self.PG_epoch_size = args.PG_epoch_size
        self.APG_epoch_size = args.APG_epoch_size

        # plotting color list
        self.color_list = [ 'lightseagreen', 'lightsalmon', \
                            'lightskyblue', 'lightpink', 'mediumpurple', \
                            'yellow', 'pink', "darkgreen"]

        # (s,a) number
        self.state_num = args.state_num
        self.action_num = args.action_num

        # sotchastic
        self.stochastic = args.stochastic

        # how many seed num to run
        self.seed_num = args.seed_num

        # discounted factor
        self.gamma = args.gamma

        # optimal policy reached by Policy Iteration
        self.optimal_policy = args.optimal_policy
        self.V_opt = args.V_opt
        self.d_rho_opt = args.d_rho_opt

        # for transition and learn
        self.initial_state_distribution = np.zeros(shape=(self.state_num), dtype=np.float64)
        self.transition_prob_1 = np.zeros(shape=(self.state_num, self.state_num, self.action_num), dtype=np.float64)
        self.transition_prob_2 = np.zeros(shape=(self.state_num, self.action_num, self.state_num), dtype=np.float64)
        self.reward = np.zeros(shape=(self.state_num, self.action_num), dtype=np.float64)
        
        for s_num in range(self.state_num):

            # initialize rho
            self.initial_state_distribution[s_num] = args.initial_state_distribution_dict[f"s{s_num+1}"]

            for s2_num in range(self.state_num):
                
                for a_num in range(self.action_num):

                    # initialize transition probability
                    self.transition_prob_1[s2_num, s_num, a_num] = args.transition_prob_dict[f"s{s_num+1}a{a_num+1}_s{s2_num+1}"]
                    self.transition_prob_2[s_num, a_num, s2_num] = args.transition_prob_dict[f"s{s_num+1}a{a_num+1}_s{s2_num+1}"]

                    # initialize reward
                    self.reward[s_num, a_num] = args.reward_dict[f"s{s_num+1}_a{a_num+1}"]
                    
        
        # state-action name including the optimal action a*
        self.state_action_pair = dict()
        for (s_num, state) in enumerate([f's{s_num+1}' for s_num in range(self.state_num)]):
            self.state_action_pair[state] = [f'a{a_num+1}' for a_num in range(self.action_num)]
            self.state_action_pair[state][self.optimal_policy[s_num]] = "a*"
        

    def plot_Summary(self, size: int, algo: str):
        
        # specify algo
        self.algo = algo

        fig = plt.figure(figsize=(16,6))
        fig.suptitle(self.algo.upper(), fontsize=20, fontdict=dict(weight='bold'), fontname='monospace')
        height, width = (2,4)

        # configuration
        self.legendsize = 10
        self.fontsize = 12
        self.offtextsize = 12
        self.ticksize = 14
        self.linewidth = 2.0
        
        # compute V(ρ)
        self.df_v_rho = self.compute_V_rho()

        # 1. policy weight
        ax_1 = plt.subplot(height, width, 1)
        self.plot_pi(ax_1, size)

        # 2. theta
        ax_2 = plt.subplot(height, width, 2, sharex=ax_1)
        self.plot_theta(ax_2, size)

        # 3. discounted state visitation distribution
        ax_3 = plt.subplot(height, width, 3, sharex=ax_1)
        self.plot_d(ax_3, size)

        # 4. Value function (V)
        ax_4 = plt.subplot(height, width, 4, sharex=ax_1)
        self.plot_V(ax_4, size)

        # 5. -log loss (-log(V* - V))
        ax_5 = plt.subplot(height, width, 5, sharex=ax_1)
        self.plot_log_loss(ax_5, size)

        # 6. Suboptimality gap (log log graph)
        ax_6 = plt.subplot(height, width, 6)
        self.plot_log_log(ax_6, size)

        # 7. Suboptimality gap
        ax_7 = plt.subplot(height, width, 7, sharex=ax_1)
        self.plot_subopt(ax_7, size)

        # 8. Mom-Grad improvement
        ax_8 = plt.subplot(height, width, 8, sharex=ax_1)
        self.plot_mom_grad(ax_8, size)

        # configuration 
        for axis in [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]:
            self.configure(axis)

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_summary_{size}.png'))
        plt.cla()
        plt.clf()
        plt.close("all")


    def plot_Value(self, size: int, algo: str):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        self.algo = algo

        # configuration
        self.legendsize = 14
        self.fontsize = 18
        self.offtextsize = 14
        self.ticksize = 16
        self.linewidth = 3.0

        # compute V(ρ)
        self.df_v_rho = self.compute_V_rho()

        # Value function (V)
        axis = plt.subplot(1, 1, 1)
        self.plot_V(axis, size)

        # configuration 
        self.configure(axis, red_text=True)
        plt.xticks([i * (size//5) for i in range(6)])

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_value_{size}.png'))
        plt.cla()
        plt.clf()
        plt.close("all")


    def plot_OneStep(self, size: int, algo: str):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        self.algo = algo

        # configuration
        self.legendsize = 14
        self.fontsize = 18
        self.offtextsize = 14
        self.ticksize = 16
        self.linewidth = 3.0

        # compute V(ρ)
        self.df_v_rho = self.compute_V_rho()

        # One step improvement
        axis = plt.subplot(1, 1, 1)
        self.plot_onestep(axis, size)

        # configuration 
        plt.yscale('symlog', linthresh=1e-8)
        self.configure(axis)
        plt.yticks([1e-2, 1e-4, 1e-6, 1e-8, 0, -1e-8])

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_one_step_{size}.png'))
        plt.cla()
        plt.clf()
        plt.close("all")


    def plot_MomGrad(self, size: int, algo: str):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        self.algo = algo

        # configuration
        self.legendsize = 14
        self.fontsize = 18
        self.offtextsize = 14
        self.ticksize = 16
        self.linewidth = 3.0

        # compute V(ρ)
        self.df_v_rho = self.compute_V_rho()

        # Mom-Grad improvement
        axis = plt.subplot(1, 1, 1)
        self.plot_mom_grad(axis, size)

        # configuration 
        self.configure(axis, loc="best", red_text=True)
        plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15])

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_mom_grad_{size}.png'))
        plt.cla()
        plt.clf()
        plt.close("all")
    

    def plot_LogLog(self, size: int):

        fig = plt.figure(figsize=(5, 4))

        # configuration
        self.legendsize = 14
        self.fontsize = 18
        self.offtextsize = 14
        self.ticksize = 16
        self.linewidth = 3.0


        axis = plt.subplot(1, 1, 1)
        
        # Plot on same graph
        for algo in ["PG", "APG"]:

            # specify algo
            self.algo = algo

            # compute V(ρ)
            self.df_v_rho = self.compute_V_rho()

            # Suboptimality gap (log log graph)
            self.plot_log_log(axis, size, clip_num=21)

        # configuration 
        self.configure(axis, sci=False)
        plt.xticks([0, 5, 10, 15, 20])
        plt.yticks([0, 5, 10, 15, 20])

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'log_log_{size}.png'))
        plt.cla()
        plt.clf()
        plt.close("all")


    def plot_pi(self, axis, size):
        
        # since 1 >= pi >= 0
        axis.set_ylim(bottom=0, top=1)

        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            self.logger(f"Plotting π(a*|{state})", title=False)

            # read the df from .parquet
            df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                columns=[f'{state}_pi_a*'])
            
            # plot
            axis.plot(df.iloc[:size][f'{state}_pi_a*'].to_numpy(), \
                        label=f"π(a*|{state}) = {round(df.iloc[size-1][f'{state}_pi_a*'], 2)}", \
                        color=color, \
                        linewidth=self.linewidth)

            # TODO
            # if seed_num != 1:
            #     ax_1.fill_between(range(1, size+1),\
            #                     df.iloc[:size][f'{state}_pi_a*'] + std_df.iloc[:size][f'{state}_pi_a*'],\
            #                     df.iloc[:size][f'{state}_pi_a*'] - std_df.iloc[:size][f'{state}_pi_a*'],\
            #                     alpha=0.25, color=color)
        
        axis.set_title("Optimal Action Policy Weight", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("π(a*|s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)


    def plot_theta(self, axis, size):
        
        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            self.logger(f"Plotting θ({state},a*)", title=False)

            # read the df from .parquet
            df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                columns=[f'{state}_theta_a*'])
            
            # plot
            axis.plot(df.iloc[:size][f'{state}_theta_a*'].to_numpy(), \
                        label=f"θ({state},a*)", \
                        color=color, \
                        linewidth=self.linewidth)
        
        axis.set_title("Optimal Action Theta", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("θ(s,a*)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_d(self, axis, size):
        
        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            self.logger(f"Plotting d_ρ(s)", title=False)

            # read the df from .parquet
            df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                columns=[f'd_rho({state})'])
            
            # plot
            axis.plot(df.iloc[:size][f'd_rho({state})'].to_numpy(), \
                        label=f"d_ρ({state}) = {round(df.iloc[size-1][f'd_rho({state})'], 2)}", \
                        color=color, \
                        linewidth=self.linewidth)
        
        axis.set_title("Discounted State Visitation Distribution", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("d_ρ(s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_V(self, axis, size):
        
        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            self.logger(f"Plotting V({state})", title=False)

            # read the df from .parquet
            df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                columns=[f'V({state})'])
            
            # plot
            axis.plot(df.iloc[:size][f'V({state})'].to_numpy(), \
                        label=f"V({state})", \
                        #  label=f"V({state}) = {round(df.iloc[size-1][f'V({state})'], 2)}"
                        color=color, \
                        linewidth=self.linewidth)
        
        # plot V(ρ)
        self.logger(f"Plotting V(ρ)", title=False)
        axis.plot(self.df_v_rho.iloc[:size]["V_theta(rho)"].to_numpy(), \
                    label=f"V(ρ)",\
                    # label=f"V(ρ) = {round(self.df_v_rho.iloc[size-1]['V_theta(rho)'], 2)}",\
                    color = "seagreen", \
                    linewidth=self.linewidth)

        axis.set_title("Value Function", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("V(s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_log_loss(self, axis, size):
        
        # log
        self.logger(f"Plotting -log(V* - V(ρ))", title=False)
        
        # plot
        axis.plot(-np.log(self.V_opt -self.df_v_rho.iloc[:size]["V_theta(rho)"].to_numpy()), \
                    label="V(ρ)", \
                    color="red" if self.algo=="APG" else "blue", \
                    linewidth=self.linewidth)
        
        axis.set_title("-Log Loss", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("-log(V* - V(ρ))", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_log_log(self, axis, size, clip_num=None):
        
        # log
        self.logger(f"Plotting -log(V*(ρ) - V(ρ))", title=False)
        
        # plot
        log_loss = -np.log(self.V_opt -self.df_v_rho.iloc[:size]["V_theta(rho)"].to_numpy())
        if clip_num:
            log_loss[log_loss >= clip_num] = np.nan
        axis.plot(np.log(range(1, size+1)), \
                    log_loss, \
                    label=self.algo, \
                    color="red" if self.algo=="APG" else "blue", \
                    linewidth=self.linewidth)
        
        axis.set_title("Sub-Optimality Gap", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("log(Time Steps)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("-log(V* - V(ρ))", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_subopt(self, axis, size):
        
        # log
        self.logger(f"Plotting V*(ρ) - V(ρ)", title=False)

        # log
        self.logger(f"Logging π(a*|s)", title=False)

        # read the df from .parquet
        df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                            columns=[f'{state}_pi_a*' for state in self.state_action_pair.keys()])
        
        # compute pre-constant mentioned in [Mei]
        d_mu_over_mu = max(self.d_rho_opt / np.clip(self.initial_state_distribution, 1e-3, None))
        inf_pi = min([min(df.iloc[:size][f'{state}_pi_a*']) for state in self.state_action_pair.keys()])
        pre_constant = pow(d_mu_over_mu, 2) \
                        * (1.0 / np.clip(min(self.initial_state_distribution), 1e-3, None)) \
                        * (16.0 * self.state_num) / (pow(inf_pi, 2) * pow(1.0 - self.gamma, 6))

        # plot
        axis.semilogy(self.V_opt -self.df_v_rho.iloc[:size]["V_theta(rho)"].to_numpy(), \
                    label=self.algo, \
                    color="red" if self.algo=="APG" else "blue", \
                    linewidth=self.linewidth)
        axis.semilogy(range(1, size+1),
                    [pre_constant / (t * t) for t in range(1, size+1)], \
                    label=f'O(1/t^2)', \
                    color = 'hotpink', \
                    linestyle="--", \
                    linewidth=self.linewidth)
        axis.semilogy(range(1, size+1),
                    [pre_constant / (t) for t in range(1, size+1)], \
                    label=f'O(1/t)', \
                    color = 'hotpink', \
                    linestyle=":", \
                    linewidth=self.linewidth)
        
        axis.set_title("Sub-Optimality Gap", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("V* - V(ρ)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_mom_grad(self, axis, size):
            
        # plot
        if self.algo == "PG":
            axis.semilogy(self.df_v_rho.iloc[1:size]["V_theta(rho)"].to_numpy() - self.df_v_rho.iloc[:size-1]["V_theta(rho)"].to_numpy(), \
                        label="Gradient", \
                        color="seagreen", \
                        linestyle=":", \
                        linewidth=self.linewidth)
        else:
            axis.semilogy(self.df_v_rho.iloc[1:size]["V_theta(rho)"].to_numpy() - self.df_v_rho.iloc[:size-1]["V_omega(rho)"].to_numpy(), \
                        label="Gradient", \
                        color="seagreen", \
                        linestyle=":", \
                        linewidth=self.linewidth)
            axis.semilogy(self.df_v_rho.iloc[2:size]["V_omega(rho)"].to_numpy() - self.df_v_rho.iloc[2:size]["V_theta(rho)"].to_numpy(), \
                        label="Momentum", \
                        color="seagreen", \
                        linewidth=self.linewidth)
        
        axis.set_title(f"One-Step Improvement", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel(f"Improvement", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def plot_onestep(self, axis, size):
            
        # plot
        axis.plot(self.df_v_rho.iloc[1:size]["V_theta(rho)"].to_numpy() - self.df_v_rho.iloc[:size-1]["V_theta(rho)"].to_numpy(), \
                    label="Diff(V)", \
                    color="seagreen", \
                    linewidth=self.linewidth)
        
        axis.set_title(f"One-Step Improvement", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel(f"V(t+1) - V(t)", fontsize=self.fontsize, fontname='monospace', labelpad=7)


    def compute_V_rho(self):
        
        # epoch size
        epoch_size = self.PG_epoch_size if self.algo == "PG" else self.APG_epoch_size

        # log
        self.logger(f"Computing V(ρ)", title=False)

        # computer V(ρ)
        df_v_rho = pd.DataFrame({'V_theta(rho)': [0]*epoch_size, 'V_omega(rho)': [0]*epoch_size}, dtype=np.float64)

        for s_num, state in enumerate(self.state_action_pair.keys()):

            # log
            self.logger(f"Logging V({state})", title=False)

            # read the df from .parquet
            if self.algo == "PG":
                df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                    columns=[f'V({state})'])
            elif self.algo == "APG":
                df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                    columns=[f'V({state})', f'V_omega({state})'])
                # accumulate V(ρ)
                df_v_rho["V_omega(rho)"] += df.iloc[:][f'V_omega({state})'] * self.initial_state_distribution[s_num]
            
            # accumulate V(ρ)
            df_v_rho["V_theta(rho)"] += df.iloc[:][f'V({state})'] * self.initial_state_distribution[s_num]
                
        
        return df_v_rho


    def configure(self, axis, sci=True, loc='best', red_text=False):
        
        # grid
        axis.grid()

        # legend
        axis.legend(loc=loc, frameon=True)
        legend = axis.legend(loc=loc, frameon=True, prop={"size":self.legendsize})
        legend.get_frame().set_linewidth(self.linewidth)
        legend.get_frame().set_edgecolor('black')

        # label
        plt.setp(axis.get_xticklabels(), fontsize=self.ticksize)
        plt.setp(axis.get_yticklabels(), fontsize=self.ticksize)
        
        # tick width
        axis.tick_params(width=2)

        # change all spines
        for ax in ['top','bottom','left','right']:
            axis.spines[ax].set_linewidth(2)

        # set x axis to scientific notation
        if sci:
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

        # offset text
        offset_text = axis.xaxis.get_offset_text()
        offset_text.set_size(self.offtextsize)
        if red_text:
            offset_text.set_color('red')
        

    def plot_Q(self, size: int, algo: str):

        fig = plt.figure(figsize=(5*self.state_num, 4))

        # specify algo
        self.algo = algo

        # configuration
        self.legendsize = 14
        self.fontsize = 18
        self.offtextsize = 14
        self.ticksize = 16
        self.linewidth = 3.0

        for num, state in enumerate(self.state_action_pair.keys()):

            axis = plt.subplot(1, self.state_num, num+1)

            for color, action in zip(self.color_list, self.state_action_pair[state]):

                # log
                self.logger(f"Plotting Q({state},{action})", title=False)

                # read the df from .parquet
                df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
                                    columns=[f'Q({state},{action})'])
                
                # plot
                axis.plot(df.iloc[:size][f'Q({state},{action})'].to_numpy(), \
                            label=f"Q({state},{action})", \
                            color=color, \
                            linewidth=self.linewidth)
                
                axis.set_title("Q-value", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
                axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
                axis.set_ylabel("Q(s,·)", fontsize=self.fontsize, fontname='monospace', labelpad=7)

                # configuration 
                self.configure(axis)
                plt.xticks([i * (size//5) for i in range(6)])

        # save plot
        plt.tight_layout()  
        plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_Q.png'))
        plt.cla()
        plt.clf()
        plt.close("all")

    # def plot_Pi(self, algo: str):

    #     fig = plt.figure(figsize=(5*self.state_num, 4))

    #     # specify algo
    #     self.algo = algo

    #     # configuration
    #     self.legendsize = 14
    #     self.fontsize = 18
    #     self.offtextsize = 14
    #     self.ticksize = 16
    #     self.linewidth = 3.0

    #     for num, state in enumerate(self.state_action_pair.keys()):

    #         axis = plt.subplot(1, self.state_num, num+1)
    #         # since 1 >= pi >= 0
    #         axis.set_ylim(bottom=0, top=1)

    #         for color, action in zip(self.color_list, self.state_action_pair[state]):

    #             # log
    #             self.logger(f"Plotting π({action}|{state})", title=False)

    #             # read the df from .parquet
    #             df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
    #                                 columns=[f'{state}_pi_{action}'])
                
    #             # plot
    #             left = 0
    #             right = 40000
    #             axis.plot(range(left, right), \
    #                         df.iloc[left:right][f'{state}_pi_{action}'].to_numpy(), \
    #                         label=f"π({action}|{state})", \
    #                         color=color, \
    #                         linewidth=self.linewidth)
                
    #             axis.set_title("Policy Weight", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
    #             axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    #             axis.set_ylabel("π(·|s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)

    #             # configuration 
    #             self.configure(axis, sci=True)
    #             plt.xticks([left + i * ((right-left)//5) for i in range(6)])

        
        

    #     # save plot
    #     plt.tight_layout()  
    #     plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_pi.png'))
    #     plt.cla()
    #     plt.clf()
    #     plt.close("all")
    

    # def plot_Theta(self, algo: str):

    #     fig = plt.figure(figsize=(5*self.state_num, 4))

    #     # specify algo
    #     self.algo = algo

    #     # configuration
    #     self.legendsize = 14
    #     self.fontsize = 18
    #     self.offtextsize = 14
    #     self.ticksize = 16
    #     self.linewidth = 3.0

    #     for num, state in enumerate(self.state_action_pair.keys()):

    #         axis = plt.subplot(1, self.state_num, num+1)

    #         for color, action in zip(self.color_list, self.state_action_pair[state]):

    #             # log
    #             self.logger(f"Plotting θ({state}, {action})", title=False)

    #             # read the df from .parquet
    #             df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
    #                                 columns=[f'{state}_theta_{action}'])
                
    #             # plot
    #             left = 1
    #             right = 40000
    #             axis.semilogy(range(left, right), \
    #                         df.iloc[left:right][f'{state}_theta_{action}'].to_numpy(), \
    #                         label=f"θ({state}, {action})", \
    #                         color=color, \
    #                         linewidth=self.linewidth)
            
    #             axis.set_title("Theta", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
    #             axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    #             axis.set_ylabel("θ(s,·)", fontsize=self.fontsize, fontname='monospace', labelpad=7)

    #             # configuration 
    #             self.configure(axis, sci=True)
    #             plt.xticks([left + i * ((right-left)//5) for i in range(6)])

    #     # save plot
    #     plt.tight_layout()  
    #     plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_theta.png'))
    #     plt.cla()
    #     plt.clf()
    #     plt.close("all")




    #     fig = plt.figure(figsize=(5*self.action_num, 4*self.state_num))

    #     # specify algo
    #     self.algo = algo

    #     # configuration
    #     self.legendsize = 14
    #     self.fontsize = 18
    #     self.offtextsize = 14
    #     self.ticksize = 16
    #     self.linewidth = 3.0

    #     # compute V(ρ)
    #     self.df_v_rho = self.compute_V_rho()

    #     # One step improvement
    #     k=1
    #     for num, state in enumerate(self.state_action_pair.keys()):
            
    #         for color, action in zip(self.color_list, self.state_action_pair[state]):
                
    #             axis = plt.subplot(self.state_num, self.action_num, k)
    #             k+=1
    #             # log
    #             self.logger(f"Plotting mom({state}, {action}), grad({state}, {action})", title=False)

    #             # read the df from .parquet
    #             df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
    #                                 columns=[f'{state}_theta_{action}', f'{state}_omega_{action}'])
                
    #             # plot
    #             left = 1
    #             right = 40000
    #             axis.semilogy(range(left, right), \
    #                         df.iloc[left:right][f'{state}_theta_{action}'].to_numpy() - df.iloc[left-1:right-1][f'{state}_omega_{action}'].to_numpy(), \
    #                         label=f"grad({state}, {action})", \
    #                         color=color, \
    #                         linestyle=":", \
    #                         linewidth=self.linewidth)
                
    #             axis.semilogy(range(left, right), \
    #                         df.iloc[left+1:right+1][f'{state}_omega_{action}'].to_numpy() - df.iloc[left+1:right+1][f'{state}_theta_{action}'].to_numpy(), \
    #                         label=f"mom({state}, {action})", \
    #                         color=color, \
    #                         linewidth=self.linewidth)
                
    #             # configuration 
    #             plt.yscale('symlog', linthresh=1e-8)
    #             self.configure(axis, sci=True)
    #             plt.xticks([left + i * ((right-left)//5) for i in range(6)])



    #             axis.set_title("M-G", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
    #             axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        
    #     # save plot
    #     plt.tight_layout()  
    #     plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_mg.png'))
    #     plt.cla()
    #     plt.clf()
    #     plt.close("all")
    
    # def plot_Adv(self, algo: str):

    #     fig = plt.figure(figsize=(5*self.state_num, 4))

    #     # specify algo
    #     self.algo = algo

    #     # configuration
    #     self.legendsize = 14
    #     self.fontsize = 18
    #     self.offtextsize = 14
    #     self.ticksize = 16
    #     self.linewidth = 3.0

    #     for num, state in enumerate(self.state_action_pair.keys()):

    #         axis = plt.subplot(1, self.state_num, num+1)

    #         for color, action in zip(self.color_list, self.state_action_pair[state]):

    #             # log
    #             self.logger(f"Plotting Adv({state}, {action})", title=False)

    #             # read the df from .parquet
    #             df = pd.read_parquet(os.path.join(self.logger.log_dir, self.algo, 'mean.parquet'), \
    #                                 columns=[f'Adv({state},{action})'])
                
    #             # plot
    #             left = 1
    #             right = 40000
    #             axis.semilogy(range(left, right), \
    #                         df.iloc[left:right][f'Adv({state},{action})'].to_numpy(), \
    #                         label=f"Adv({state},{action})", \
    #                         color=color, \
    #                         linewidth=self.linewidth)
            
    #             axis.set_title("Theta", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
    #             axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    #             axis.set_ylabel("Adv(s,·)", fontsize=self.fontsize, fontname='monospace', labelpad=7)

    #             # configuration 
    #             plt.yscale('symlog', linthresh=1e-8)
    #             self.configure(axis, sci=True)
    #             plt.xticks([left + i * ((right-left)//5) for i in range(6)])

    #     # save plot
    #     plt.tight_layout()  
    #     plt.savefig(os.path.join(self.logger.log_dir, f'{self.algo}', f'{self.algo}_adv.png'))
    #     plt.cla()
    #     plt.clf()
    #     plt.close("all")