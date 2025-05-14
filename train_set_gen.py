import os
import pandas as pd
import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from conway_lib import ConwayGame
conway_game=ConwayGame()


def generate_sets(A=100, N=32, I=2, Toroidal=False, save_folder='data'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(save_folder, exist_ok=True)

    game = ConwayGame(width=N, height=N, grid_size=1)

    start_time = time.time()
    
    start = 0 # order param of the frist sample
    end = 1 # order param of the last sample
    train_data = game.generate_sets(A=A, N=N, I=I, s=start, e=end)

    data = []
    for index in range(A):
        states = train_data[index]
        data.append(states)
        
    columns = [f'State {i + 1}' for i in range(I)]
    df = pd.DataFrame(data, columns=columns)
    print(df.shape)
    
    toroidal_str = 'toroidal' if Toroidal else 'non_toroidal'
    save_path = os.path.join(save_folder, f'conway_states_{start}_{end}_{A}by{N}by{N}by{I}_{toroidal_str}_{timestamp}.csv')
    df.to_csv(save_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CSV generation time: {elapsed_time} seconds")


# Generate patterns
def generate_test_sets(N=32, I=10, Toroidal=True, save_folder='patterns'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(save_folder, exist_ok=True)

    game = ConwayGame(toroidal=Toroidal, width=N, height=N, grid_size=1)

    start_time = time.time()

    test_data = []
    # order_params = [0, 0.25, 0.5, 0.75, 1]
    # for op in order_params:
    #     game.randomize_grid_uniform(op)
    #     game.run(num_iterations=I)

    #     metagrid = game.metagrid
    #     states = [game.get_state_as_string(metagrid[i]) for i in range(I)]
    #     test_data.append(states)
    #     animate_game(game, I, f'order_param_{op}.gif')

    specific_patterns = ["gliders", "cloverleaf", "hammerhead_spaceship", "blinkers", "r_pentomino"]
    for pattern in specific_patterns:
        game.initialize_pattern(pattern)
        game.run(num_iterations=I)
        metagrid = game.metagrid
        states = [game.get_state_as_string(metagrid[i]) for i in range(I)]
        test_data.append(states)
        animate_game(game, I, f'{pattern}.gif')

    columns = [f'State {i + 1}' for i in range(I)]
    df = pd.DataFrame(test_data, columns=columns)
    print(df.shape)
    
    save_path = os.path.join(save_folder, f'conway_test_states_{N}by{N}_{timestamp}.csv')
    df.to_csv(save_path, index=False)
    print(f"Test set saved to: {save_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CSV generation time: {elapsed_time} seconds")


def generate_spaceships(N=8, I=10, Toroidal=True, save_folder='patterns'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(save_folder, exist_ok=True)

    game = ConwayGame(toroidal=Toroidal, width=N, height=N, grid_size=1)

    start_time = time.time()

    test_data = []
    specific_patterns = ["glider", "lwss", "mwss", "hwss"]
    for pattern in specific_patterns:
        game.initialize_pattern(pattern)
        game.run(num_iterations=I)
        metagrid = game.metagrid
        states = [game.get_state_as_string(metagrid[i]) for i in range(I)]
        test_data.append(states)
        animate_game(game, I, f'{pattern}.gif')

    columns = [f'State {i + 1}' for i in range(I)]
    df = pd.DataFrame(test_data, columns=columns)
    print(df.shape)
    
    save_path = os.path.join(save_folder, f'conway_spaceships_{N}by{N}_{timestamp}.csv')
    df.to_csv(save_path, index=False)
    print(f"Test set saved to: {save_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CSV generation time: {elapsed_time} seconds")


def animate_game(game, num_iterations, filename, interval=200):
    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        ax.imshow(game.metagrid[frame], cmap='binary')
        ax.set_title(f"Iteration {frame+1}")
    ani = FuncAnimation(fig, update, frames=num_iterations, interval=interval, repeat=False)
    writer = PillowWriter(fps=5)  # Save as GIF
    ani.save(filename, writer=writer)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Conway sets')
    parser.add_argument('--num_sim', type=int, default=10000, help='Number of simulations to generate')
    parser.add_argument('--img_size', type=int, default=32, help='Grid size')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of simulation steps to run')
    args = parser.parse_args()

    generate_sets(A=args.num_sim, N=args.img_size, I=args.num_steps, Toroidal=True, save_folder='data')
    # generate_test_sets()
    # generate_spaceships(N=args.img_size, I=args.num_steps, Toroidal=True, save_folder='patterns')
