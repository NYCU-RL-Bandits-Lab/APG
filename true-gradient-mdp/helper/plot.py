# package
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')                      # backend
import matplotlib.pyplot as plt
from loguru import logger

# -------------- Plotting --------------
def configure(
        axis,
        legendsize=14,
        offtextsize=14,
        ticksize=16,
        linewidth=3.0,
        sci=True,
        loc='best',
        red_text=False,
        grid=True,
    ):
        
    # grid
    if grid:
        axis.grid()

    # legend
    if legendsize:
        axis.legend(loc=loc, frameon=True)
        legend = axis.legend(loc=loc, frameon=True, prop={"size":legendsize})
        legend.get_frame().set_linewidth(linewidth)
        legend.get_frame().set_edgecolor('black')

    # label
    plt.setp(axis.get_xticklabels(), fontsize=ticksize)
    plt.setp(axis.get_yticklabels(), fontsize=ticksize)
    
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
    offset_text.set_size(offtextsize)
    if red_text:
        offset_text.set_color('red')


class Plotter:

    def __init__(self, args: object, log_dir: str):

        # log_dir
        self.log_dir = log_dir

        # algo
        self.algo = args.algo

        # epoch size
        self.epoch = args.epoch

        # plotting color list
        self.color_list = [
            'lightseagreen',
            'lightsalmon',
            'lightskyblue',
            'lightpink',
            'mediumpurple',
            'yellow',
            'pink',
            'darkgreen',
        ]

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
        
        # algo color
        self.color_dict = {
            'APG': 'blue',
            'NAPG': 'c',
            'APG_adaptive': 'c',
            'PG': 'red',
            'PG_heavy_ball': 'darkorange',
            'PG_adam': 'mediumpurple',
            'NPG': 'darkgreen',
        }

        # algo name
        self.name_dict = {
            'APG': 'APG (Ours)',
            'NAPG': 'NAPG (Ours)',
            'APG_adaptive': 'APG + exp-growing lr',
            'PG': 'PG',
            'PG_heavy_ball': 'HBPG',
            'PG_adam': 'PG + Adam',
            'NPG': 'NPG + constant lr',
        }
        
    def plot_summary(self, size: int):

        # specify algo
        img_path = os.path.join(self.log_dir, f'{self.algo}_summary_{size}.png')
        logger.info(f"[{self.algo}] Plotting summary at: {img_path}")

        # fig title
        fig = plt.figure(figsize=(16,6))
        fig.suptitle(
            self.algo.upper() if (self.seed_num == 1) else f'{self.algo.upper()} (num_seed = {self.seed_num})',
            fontsize=20,
            fontdict=dict(weight='bold'),
            fontname='monospace',
        )
        height, width = (2,4)

        # configuration
        self.fontsize = 12
        self.linewidth = 2.0
        
        # 1. policy weight
        ax_1 = plt.subplot(height, width, 1)
        self._plot_pi(ax_1, size)

        # 2. theta
        ax_2 = plt.subplot(height, width, 2, sharex=ax_1)
        self._plot_theta(ax_2, size)

        # 3. discounted state visitation distribution
        ax_3 = plt.subplot(height, width, 3, sharex=ax_1)
        self._plot_d(ax_3, size)

        # 4. Value function (V)
        ax_4 = plt.subplot(height, width, 4, sharex=ax_1)
        self._plot_value(ax_4, size)

        # 5. Suboptimality gap (log graph)
        ax_5 = plt.subplot(height, width, 5, sharex=ax_1)
        self._plot_log(ax_5, size)

        # 6. Suboptimality gap (log log graph)
        ax_6 = plt.subplot(height, width, 6)
        self._plot_log_log(ax_6, size)

        # 7. Suboptimality gap
        ax_7 = plt.subplot(height, width, 7, sharex=ax_1)
        self._plot_subopt(ax_7, size)

        # 8. Mom-Grad improvement
        ax_8 = plt.subplot(height, width, 8, sharex=ax_1)
        self._plot_mom_grad(ax_8, size)

        # configuration 
        for axis in [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6, ax_7, ax_8]:
            configure(
                axis,
                legendsize=10,
                offtextsize=12,
                ticksize=14,
                linewidth=2.0,
                sci=True,
                loc='best',
                red_text=False,
            )

        # save plot
        plt.tight_layout()  
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path

    def plot_value(self, size: int, algos: list = None, state_wise=True):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        img_path = os.path.join(self.log_dir, f'value_{size}.png')
        logger.info(f"[{self.algo}] Plotting value function at: {img_path}")

        # configuration
        self.fontsize = 18
        self.linewidth = 3.0

        axis = plt.subplot(1, 1, 1)

        if not algos:
            algos = [self.algo]
        for algo in algos:

            # specify algo
            self.algo = algo

            # Value function
            self._plot_value(axis, size, state_wise)

        # configuration 
        configure(
            axis,
            sci=True,
            loc='best',
            red_text=True,
        )
        plt.xticks([i * (size//5) for i in range(6)])

        # save plot
        plt.tight_layout()  
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path

    def plot_one_step(self, size: int):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        img_path = os.path.join(self.log_dir, f'{self.algo}', f'{self.algo}_one_step_{size}.png')
        logger.info(f"[{self.algo}] Plotting one step improvemnt at: {img_path}")

        # configuration
        self.fontsize = 18
        self.linewidth = 3.0

        # One step improvement
        axis = plt.subplot(1, 1, 1)
        self.plot_onestep_(axis, size)

        # configuration 
        plt.yscale('symlog', linthresh=1e-8)
        configure(
            axis,
            sci=True,
            loc='best',
            red_text=False,
        )
        plt.yticks([1e-2, 1e-4, 1e-6, 1e-8, 0, -1e-8])

        # save plot
        plt.tight_layout()  
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path

    def plot_mom_grad(self, size: int):

        fig = plt.figure(figsize=(5, 4))

        # specify algo
        img_path = os.path.join(self.log_dir, f'{self.algo}', f'{self.algo}_mom_grad_{size}.png')
        logger.info(f"[{self.algo}] Plotting mom-grad at: {img_path}")

        # configuration
        self.fontsize = 18
        self.linewidth = 2.0

        # Mom-Grad improvement
        axis = plt.subplot(1, 1, 1)
        self._plot_mom_grad(axis, size)

        # configuration 
        configure(
            axis,
            sci=True,
            loc='best',
            red_text=True,
        )
        plt.yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15])

        # save plot
        plt.tight_layout()
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path

    def plot_log_log(self, size: int, algos: list = None):

        fig = plt.figure(figsize=(5, 4))

        img_path = os.path.join(self.log_dir, f'log_log_{size}.png')
        logger.info(f"[{self.algo}] Plotting log-log plot at: {img_path}")

        # configuration
        self.fontsize = 18
        self.linewidth = 3.0

        axis = plt.subplot(1, 1, 1)
        
        # Plot on same graph
        if not algos:
            algos = [self.algo]
        for algo in algos:

            # specify algo
            self.algo = algo

            # Suboptimality gap (log log graph)
            self._plot_log_log(axis, size, clip_num=40)
                
        # configuration 
        configure(
            axis,
            sci=False,
            loc='upper left',
            red_text=True,
        )
        plt.xticks([0, 5, 10, 15, 20])
        plt.yticks([0, 10, 20, 30, 40])

        # save plot
        plt.tight_layout()  
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path
    
    def plot_log(self, size: int, algos: list = None):

        fig = plt.figure(figsize=(5, 4))

        # configuration
        self.fontsize = 18

        img_path = os.path.join(self.log_dir, f'log_{size}.png')
        logger.info(f"[{self.algo}] Plotting log plot at: {img_path}")

        # configuration
        self.fontsize = 18
        self.linewidth = 3.0

        axis = plt.subplot(1, 1, 1)
        
        # Plot on same graph
        if not algos:
            algos = [self.algo]
        for algo in algos:

            # specify algo
            self.algo = algo

            # Suboptimality gap (log log graph)
            self._plot_log(axis, size, clip_num=40)
                
        # configuration 
        configure(
            axis,
            sci=False,
            loc='upper left',
            red_text=True,
        )
        # plt.xticks([0, 5, 10, 15, 20])
        # plt.yticks([0, 10, 20, 30, 40])

        # save plot
        plt.tight_layout()  
        plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close("all")

        return img_path

    def _plot_pi(self, axis, size):
        
        # since 1 >= pi >= 0
        axis.set_ylim(bottom=0, top=1)

        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            logger.debug(f"[{self.algo}] Plotting π(a*|{state})")

            # read the df from .parquet
            df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'mean.parquet'),
                columns=['timestep', f'{state}_pi_a*'],
            )
            df = df[df.timestep <= size]

            # plot
            axis.plot(
                df.timestep,
                df[f'{state}_pi_a*'].to_numpy(),
                label=f"π(a*|{state}) = {round(df.iloc[-1][f'{state}_pi_a*'], 2)}",
                color=color,
                linewidth=self.linewidth,
            )

            # std
            if self.seed_num != 1:

                # read the df from .parquet
                std_df = pd.read_parquet(
                    os.path.join(self.log_dir, self.algo, 'std.parquet'),
                    columns=['timestep', f'{state}_pi_a*'],
                )
                std_df = std_df[std_df.timestep <= size]

                # plot
                axis.fill_between(
                    std_df.timestep,
                    df[f'{state}_pi_a*'] + std_df[f'{state}_pi_a*'],
                    df[f'{state}_pi_a*'] - std_df[f'{state}_pi_a*'],
                    alpha=0.25,
                    color=color,
                )

        axis.set_title("Optimal Action Policy Weight", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("π(a*|s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)


    def _plot_theta(self, axis, size):
        
        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            logger.debug(f"[{self.algo}] Plotting θ({state},a*)")

            # read the df from .parquet
            df = pd.read_parquet(os.path.join(self.log_dir, self.algo, 'mean.parquet'), \
                                columns=['timestep', f'{state}_theta_a*'])
            df = df[df.timestep <= size]
            
            # plot
            axis.plot(
                df.timestep,
                df[f'{state}_theta_a*'].to_numpy(),
                label=f"θ({state},a*)",
                color=color,
                linewidth=self.linewidth,
            )

            # std
            if self.seed_num != 1:

                # read the df from .parquet
                std_df = pd.read_parquet(
                    os.path.join(self.log_dir, self.algo, 'std.parquet'),
                    columns=['timestep', f'{state}_theta_a*'],
                )
                std_df = std_df[std_df.timestep <= size]

                # plot
                axis.fill_between(
                    std_df.timestep,
                    df[f'{state}_theta_a*'] + std_df[f'{state}_theta_a*'],
                    df[f'{state}_theta_a*'] - std_df[f'{state}_theta_a*'],
                    alpha=0.25,
                    color=color,
                )
        
        axis.set_title("Optimal Action Theta", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("θ(s,a*)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def _plot_d(self, axis, size):
        
        for color, state in zip(self.color_list, self.state_action_pair.keys()):

            # log
            logger.debug(f"[{self.algo}] Plotting d_ρ(s)")

            # read the df from .parquet
            df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'mean.parquet'),
                columns=['timestep', f'd_rho({state})']
            )
            df = df[df.timestep <= size]
            
            # plot
            axis.plot(
                df.timestep,
                df[f'd_rho({state})'].to_numpy(),
                label=f"d_ρ({state}) = {round(df.iloc[-1][f'd_rho({state})'], 2)}",
                color=color,
                linewidth=self.linewidth
            )

            # std
            if self.seed_num != 1:

                # read the df from .parquet
                std_df = pd.read_parquet(
                    os.path.join(self.log_dir, self.algo, 'std.parquet'),
                    columns=['timestep', f'd_rho({state})'],
                )
                std_df = std_df[std_df.timestep <= size]

                # plot
                axis.fill_between(
                    std_df.timestep, 
                    df[f'd_rho({state})'] + std_df[f'd_rho({state})'],
                    df[f'd_rho({state})'] - std_df[f'd_rho({state})'],
                    alpha=0.25,
                    color=color,
                )
        
        axis.set_title("Discounted State Visitation Distribution", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("d_ρ(s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def _plot_value(self, axis, size, state_wise=True):
        
        if state_wise:
            
            for color, state in zip(self.color_list, self.state_action_pair.keys()):

                # log
                logger.debug(f"[{self.algo}] Plotting V({state})")

                # read the df from .parquet
                df = pd.read_parquet(
                    os.path.join(self.log_dir, self.algo, 'mean.parquet'),
                    columns=['timestep', f'V({state})', 'V(rho)'],
                )
                df = df[df.timestep <= size]
                
                # plot
                axis.plot(
                    df.timestep,
                    df[f'V({state})'].to_numpy(),
                    label=f"V({state})",
                    #  label=f"V({state}) = {round(df.iloc[df.shape[0]-1][f'V({state})'], 2)}"
                    color=color,
                    linewidth=self.linewidth,
                )
                
                # std
                if self.seed_num != 1:

                    # read the df from .parquet
                    std_df = pd.read_parquet(
                        os.path.join(self.log_dir, self.algo, 'std.parquet'),
                        columns=['timestep', f'V({state})', 'V(rho)'],
                    )
                    std_df = std_df[std_df.timestep <= size]

                    # plot
                    axis.fill_between(
                        std_df.timestep, 
                        df[f'V({state})'] + std_df[f'V({state})'],
                        df[f'V({state})'] - std_df[f'V({state})'],
                        alpha=0.25,
                        color=color,
                    )
        
        # plot V(ρ)
        logger.debug(f"[{self.algo}] Plotting V(ρ)")

        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(self.log_dir, self.algo, 'mean.parquet'),
            columns=['timestep', 'V(rho)'],
        )
        df = df[df.timestep <= size]
        axis.plot(
            df.timestep,
            df[f'V(rho)'].to_numpy(),
            label=f"V(ρ)",
            # label=f"V(ρ) = {round(self.df_v_rho.iloc[df.shape[0]-1]['V_theta(rho)'], 2)}",
            color=self.color_dict[self.algo], 
            linewidth=self.linewidth,
        )

        # std
        if self.seed_num != 1:

            # read the df from .parquet
            std_df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'std.parquet'),
                columns=['timestep', 'V(rho)'],
            )
            std_df = std_df[std_df.timestep <= size]

            # plot
            axis.fill_between(
                std_df.timestep, 
                df[f'V(rho)'] + std_df[f'V(rho)'],
                df[f'V(rho)'] - std_df[f'V(rho)'],
                alpha=0.25,
                color=self.color_dict[self.algo],
            )

        axis.set_title("Value Function", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("V(s)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    

    def _plot_log(self, axis, size, clip_num=None):
        
        # log
        logger.debug(f"[{self.algo}] Plotting -log(V*(ρ) - V(ρ))")

        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(self.log_dir, self.algo, 'mean.parquet'),
            columns=['timestep', '-log(V_opt-V(rho))'],
        )
        df = df[df.timestep <= size]
        
        # plot
        log_loss = df["-log(V_opt-V(rho))"].to_numpy()
        if clip_num:
            log_loss[log_loss >= clip_num] = np.nan
        axis.plot(
            df.timestep,
            log_loss,
            label=self.name_dict[self.algo],
            color=self.color_dict[self.algo],
            linewidth=self.linewidth,
        )

        # std
        if self.seed_num != 1:

            # read the df from .parquet
            std_df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'std.parquet'),
                columns=['timestep', f'-log(V_opt-V(rho))'],
            )
            std_df = std_df[std_df.timestep <= size]

            # plot
            axis.fill_between(
                std_df.timestep, 
                df[f'-log(V_opt-V(rho))'] + std_df[f'-log(V_opt-V(rho))'],
                df[f'-log(V_opt-V(rho))'] - std_df[f'-log(V_opt-V(rho))'],
                alpha=0.25,
                color=self.color_dict[self.algo],
            )
        
        axis.set_title("Sub-Optimality Gap", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("-log(V* - V(ρ))", fontsize=self.fontsize, fontname='monospace', labelpad=7)

    def _plot_log_log(self, axis, size, clip_num=None):
        
        # log
        logger.debug(f"[{self.algo}] Plotting -log(V*(ρ) - V(ρ))")
        
        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(self.log_dir, self.algo, 'mean.parquet'),
            columns=['timestep', 'log(timestep)', '-log(V_opt-V(rho))'],
        )
        df = df[df.timestep <= size]

        # plot
        log_loss = df[f'-log(V_opt-V(rho))'].to_numpy()
        if clip_num:
            log_loss[log_loss >= clip_num] = np.nan
        axis.plot(
            df[f'log(timestep)'].to_numpy(),
            log_loss,
            label=self.name_dict[self.algo],
            color=self.color_dict[self.algo],
            linewidth=self.linewidth,
        )

        # std
        if self.seed_num != 1:

            # read the df from .parquet
            std_df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'std.parquet'),
                columns=['log(timestep)', f'-log(V_opt-V(rho))'],
            )
            std_df = std_df[std_df['log(timestep)'] <= size]

            # plot
            axis.fill_between(
                std_df[f'log(timestep)'].to_numpy(), 
                df[f'-log(V_opt-V(rho))'] + std_df[f'-log(V_opt-V(rho))'],
                df[f'-log(V_opt-V(rho))'] - std_df[f'-log(V_opt-V(rho))'],
                alpha=0.25,
                color=self.color_dict[self.algo],
            )
        
        axis.set_title("Sub-Optimality Gap", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("log(Time Steps)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("-log(V* - V(ρ))", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    
    def _plot_subopt(self, axis, size):
        
        # log
        logger.debug(f"[{self.algo}] Plotting V*(ρ) - V(ρ)")

        # log
        logger.debug(f"[{self.algo}] Logging π(a*|s)")

        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(
                self.log_dir, 
                self.algo, 
                'mean.parquet'
            ),
            columns=['timestep'] + [f'{state}_pi_a*' for state in self.state_action_pair.keys()] + ['V(rho)']
        )
        df = df[df.timestep <= size]
        
        # compute pre-constant mentioned in [Mei]
        d_mu_over_mu = max(self.d_rho_opt / np.clip(self.initial_state_distribution, 1e-3, None))
        inf_pi = min([min(df[f'{state}_pi_a*']) for state in self.state_action_pair.keys()])
        pre_constant = pow(d_mu_over_mu, 2) \
                        * (1.0 / np.clip(min(self.initial_state_distribution), 1e-3, None)) \
                        * (16.0 * self.state_num) / (pow(inf_pi, 2) * pow(1.0 - self.gamma, 6))

        # plot
        axis.semilogy(
            df.timestep,
            self.V_opt - df["V(rho)"].to_numpy(),
            label=self.algo,
            color=self.color_dict[self.algo],
            linewidth=self.linewidth,
        )
        axis.semilogy(
            range(1, size+1),
            [pre_constant / (t * t) for t in range(1, size+1)],
            label=f'O(1/t^2)',
            color = 'hotpink',
            linestyle="--",
            linewidth=self.linewidth,
        )
        axis.semilogy(
            range(1, size+1),
            [pre_constant / (t) for t in range(1, size+1)],
            label=f'O(1/t)',
            color = 'hotpink',
            linestyle=":",
            linewidth=self.linewidth,
        )

        # std
        if self.seed_num != 1:

            # read the df from .parquet
            std_df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'std.parquet'),
                columns=['timestep', f'V(rho)'],
            )
            std_df = std_df[std_df.timestep <= size]

            # plot
            axis.fill_between(
                std_df.timestep, 
                (self.V_opt - df["V(rho)"].to_numpy()) + std_df['V(rho)'],
                (self.V_opt - df["V(rho)"].to_numpy()) - std_df['V(rho)'],
                alpha=0.25,
                color=self.color_dict[self.algo],
            )
        
        axis.set_title("Sub-Optimality Gap", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel("V* - V(ρ)", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    
    def _plot_mom_grad(self, axis, size):
            
        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(self.log_dir, self.algo, 'mean.parquet'),
            columns=['timestep', 'V(rho)'] if self.algo in ["PG", "NPG", "PG_adam"] else ['timestep', 'V(rho)', 'V_omega(rho)'],
        )
        df = df[df.timestep <= size]

        # plot
        if self.algo in ["PG", "NPG", "PG_adam"]:
            
            axis.semilogy(
                df.iloc[1:df.shape[0]]["timestep"],
                df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V(rho)"].to_numpy(),
                label="Gradient",
                color="seagreen",
                linestyle=":",
                linewidth=self.linewidth,
            )
        else:
            axis.semilogy(
                df.iloc[1:df.shape[0]]["timestep"],
                df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V_omega(rho)"].to_numpy(),
                label="Gradient",
                color="seagreen",
                linestyle=":",
                linewidth=self.linewidth,
            )
            axis.semilogy(
                df.iloc[1:df.shape[0]]["timestep"],
                df.iloc[1:df.shape[0]]["V_omega(rho)"].to_numpy() - df.iloc[1:df.shape[0]]["V(rho)"].to_numpy(),
                label="Momentum",
                color="seagreen",
                linewidth=self.linewidth,
            )
        
        # std
        if self.seed_num != 1:

            # read the df from .parquet
            std_df = pd.read_parquet(
                os.path.join(self.log_dir, self.algo, 'std.parquet'),
                columns=['timestep', 'V(rho)'] if self.algo in ["PG", "NPG", "PG_adam"] else ['timestep', 'V(rho)', 'V_omega(rho)'],
            )
            std_df = std_df[std_df.timestep <= size]

            if self.algo in ["PG", "NPG", "PG_adam"]:
            
                axis.fill_between(
                    df.iloc[1:df.shape[0]]["timestep"],
                    (df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V(rho)"].to_numpy()) + (std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy() - std_df.iloc[:std_df.shape[0]-1]["V(rho)"].to_numpy()),
                    (df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V(rho)"].to_numpy()) - (std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy() - std_df.iloc[:std_df.shape[0]-1]["V(rho)"].to_numpy()),
                    color="seagreen",
                    linestyle=":",
                    linewidth=self.linewidth,
                )
            else:
                axis.fill_between(
                    df.iloc[1:df.shape[0]]["timestep"],
                    (df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V_omega(rho)"].to_numpy()) + (std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy() - std_df.iloc[:std_df.shape[0]-1]["V_omega(rho)"].to_numpy()),
                    (df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V_omega(rho)"].to_numpy()) - (std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy() - std_df.iloc[:std_df.shape[0]-1]["V_omega(rho)"].to_numpy()),
                    color="seagreen",
                    linestyle=":",
                    linewidth=self.linewidth,
                )
                axis.fill_between(
                    df.iloc[1:df.shape[0]]["timestep"],
                    (df.iloc[1:df.shape[0]]["V_omega(rho)"].to_numpy() - df.iloc[1:df.shape[0]]["V(rho)"].to_numpy()) + (std_df.iloc[1:std_df.shape[0]]["V_omega(rho)"].to_numpy() - std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy()),
                    (df.iloc[1:df.shape[0]]["V_omega(rho)"].to_numpy() - df.iloc[1:df.shape[0]]["V(rho)"].to_numpy()) - (std_df.iloc[1:std_df.shape[0]]["V_omega(rho)"].to_numpy() - std_df.iloc[1:std_df.shape[0]]["V(rho)"].to_numpy()),
                    color="seagreen",
                    linewidth=self.linewidth,
                )

        axis.set_title(f"One-Step Improvement", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel(f"Improvement", fontsize=self.fontsize, fontname='monospace', labelpad=7)
    
    def plot_onestep_(self, axis, size):
        
        # read the df from .parquet
        df = pd.read_parquet(
            os.path.join(self.log_dir, self.algo, 'mean.parquet'),
            columns=['timestep', 'V(rho)'],
        )
        df = df[df.timestep <= size]

        # plot
        axis.plot(
            df.iloc[:df.shape[0]-1]["timestep"],
            df.iloc[1:df.shape[0]]["V(rho)"].to_numpy() - df.iloc[:df.shape[0]-1]["V(rho)"].to_numpy(), 
            label="Diff(V)", 
            color="seagreen", 
            linewidth=self.linewidth,
        )
        
        axis.set_title(f"One-Step Improvement", fontsize=self.fontsize, fontdict=dict(weight='bold'), fontname='monospace', pad=12)
        axis.set_xlabel("Time Steps", fontsize=self.fontsize, fontname='monospace', labelpad=7)
        axis.set_ylabel(f"V(t+1) - V(t)", fontsize=self.fontsize, fontname='monospace', labelpad=7)