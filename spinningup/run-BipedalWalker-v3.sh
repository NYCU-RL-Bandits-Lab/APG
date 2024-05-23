#!/usr/bin/env bash
source /home/m-ru/anaconda3/bin/activate
conda activate spinningup
env=BipedalWalker-v3
epoch=6000
hid=32
max_ep_len=1600
for seed in {0..4}
do
	python -m spinup.run vpg_sgd_pytorch \
		--hid "[$hid,$hid]" \
		--env $env \
		--exp_name $env-sgd \
		--gamma 0.999 \
		--max_ep_len $max_ep_len \
		--pi_lr 0.03 \
		--seed $seed \
		--epochs $epoch
	python -m spinup.run vpg_sgd_nesterov_pytorch \
		--hid "[$hid,$hid]" \
		--env $env \
		--exp_name $env-apg \
		--gamma 0.999 \
		--max_ep_len $max_ep_len \
		--pi_lr 0.0007 \
		--seed $seed \
		--epochs $epoch
	python -m spinup.run trpo \
			--hid "[$hid,$hid]" \
			--env $env \
			--exp_name $env-npg \
			--gamma 0.999 \
			--max_ep_len $max_ep_len \
			--seed $seed \
			--algo npg \
			--epochs $epoch
	
done