#!/bin/bash

# python train_gpt.py --num_frames 256 --t_skip 7 --train_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250525_012708.csv --val_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250525_014436.csv --seed 0 --loss_beta 2.8 --train_steps 50000

# python train_gpt.py --num_frames 512 --t_skip 15 --train_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250525_020248.csv --val_csv data/conway_states_0_1_10000by32by32by512_toroidal_20250525_023945.csv --seed 42 --loss_beta 4.0 --train_steps 50000

# python train_gpt.py --num_frames 256 --t_skip 7 --train_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250525_012708.csv --val_csv data/conway_states_0_1_10000by32by32by256_toroidal_20250525_014436.csv --seed 42 --loss_beta 2.8 --train_steps 50000

# python train_gpt.py --num_frames 32 --t_skip 0 --train_csv data/conway_states_0_1_10000by32by32by32_toroidal_20250524_172900.csv --val_csv data/conway_states_0_1_10000by32by32by32_toroidal_20250524_173336.csv --seed 42 --loss_beta 2.0 --train_steps 30000

# python train_gpt.py --num_frames 128 --t_skip 3 --train_csv data/conway_states_0_1_10000by32by32by128_toroidal_20250524_205237.csv --val_csv data/conway_states_0_1_10000by32by32by128_toroidal_20250524_210123.csv --seed 42 --loss_beta 2.0 --train_steps 50000

# python train_gpt.py --num_frames 192 --t_skip 5 --train_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250527_151201.csv --val_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250527_153501.csv --seed 0 --loss_beta 2.0 --train_steps 50000

# python train_gpt.py --num_frames 192 --t_skip 5 --train_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250527_151201.csv --val_csv data/conway_states_0_1_10000by32by32by192_toroidal_20250527_153501.csv --seed 42 --loss_beta 2.0 --train_steps 50000

# python train_gpt.py --num_frames 1024 --t_skip 31 --train_csv data/conway_states_0_1_10000by32by32by1024_toroidal_20250527_011603.csv --val_csv data/conway_states_0_1_10000by32by32by1024_toroidal_20250527_040134.csv --seed 42 --loss_beta 6.0 --train_steps 50000

# python train_gpt.py --num_frames 1024 --t_skip 31 --train_csv data/conway_states_0_1_10000by32by32by1024_toroidal_20250527_011603.csv --val_csv data/conway_states_0_1_10000by32by32by1024_toroidal_20250527_040134.csv --seed 0 --loss_beta 6.0 --train_steps 50000


# python train_set_gen.py --img_size 32 --num_steps 96

# python train_set_gen.py --img_size 32 --num_steps 96

python train_gpt.py --num_frames 64 --t_skip 1 --train_csv data/conway_states_0.4_0.4_10000by32by32by64_toroidal_20250602_161505.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by64_toroidal_20250602_161935.csv --seed 42 --loss_beta 1.0 --train_steps 50000


# python train_set_gen.py --img_size 32 --num_steps 256

# python train_gpt.py --num_frames 32 --t_skip 0 --train_csv data/conway_states_0.4_0.4_10000by32by32by32_toroidal_20250528_175553.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by32_toroidal_20250528_175810.csv --seed 42 --loss_beta 1.0 --train_steps 30000

# python train_gpt.py --num_frames 128 --t_skip 3 --train_csv data/conway_states_0.4_0.4_10000by32by32by128_toroidal_20250528_180925.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by128_toroidal_20250528_181825.csv --seed 42 --loss_beta 2.3 --train_steps 50000

# python train_gpt.py --num_frames 160 --t_skip 4 --train_csv data/conway_states_0.4_0.4_10000by32by32by160_toroidal_20250528_182730.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by160_toroidal_20250528_183840.csv --seed 42 --loss_beta 2.3 --train_steps 100000

# python train_gpt.py --num_frames 192 --t_skip 5 --train_csv data/conway_states_0.4_0.4_10000by32by32by192_toroidal_20250528_185007.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by192_toroidal_20250528_191729.csv --seed 42 --loss_beta 2.3 --train_steps 100000

# python train_gpt.py --num_frames 256 --t_skip 7 --train_csv data/conway_states_0.4_0.4_10000by32by32by256_toroidal_20250528_193056.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by256_toroidal_20250528_194823.csv --seed 42 --loss_beta 2.8 --train_steps 100000

# python train_gpt.py --num_frames 512 --t_skip 15 --train_csv data/conway_states_0.4_0.4_10000by32by32by512_toroidal_20250531_232832.csv --val_csv data/conway_states_0.4_0.4_10000by32by32by512_toroidal_20250601_000456.csv --seed 42 --loss_beta 3.8 --train_steps 100000
