#!/bin/bash

# python train_gpt.py --img_size 32 --num_frames 64 --train_csv data/conway_states_0_1_10000by32by32by64_toroidal_20250509_183710.csv --val_csv data/conway_states_0_1_10000by32by32by64_toroidal_20250509_183938.csv --patches_per_dim 8 --t_skip 1 --batch_size 16

# python train_gpt.py --img_size 32 --num_frames 128 --train_csv data/conway_states_0_1_10000by32by32by128_toroidal_20250509_192528.csv --val_csv data/conway_states_0_1_10000by32by32by128_toroidal_20250509_193024.csv --patches_per_dim 8 --t_skip 3 --batch_size 16

# python train_gpt.py --img_size 32 --num_frames 192 --train_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250511_184139.csv --val_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250511_184927.csv --patches_per_dim 8 --t_skip 5 --batch_size 16 --loss_beta 2.0

python train_gpt.py --img_size 32 --num_frames 256 --train_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250509_193519.csv --val_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250509_194632.csv --patches_per_dim 8 --t_skip 7 --batch_size 16 --loss_beta 3.0

python train_gpt.py --img_size 32 --num_frames 512 --train_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250509_195635.csv --val_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250510_041838.csv --patches_per_dim 8 --t_skip 15 --batch_size 16 --loss_beta 3.0

python train_gpt.py --img_size 32 --num_frames 512 --train_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250509_195635.csv --val_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250510_041838.csv --patches_per_dim 8 --t_skip 15 --batch_size 16 --loss_beta 4.0
