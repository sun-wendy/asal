import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from conway_lib.game import ConwayGame

def generate_test_sets(N=32, I=10, Toroidal=True, save_folder='patterns'):
    # Create a timestamp for the current date and time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Ensure that the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Create an instance of ConwayGame
    game = ConwayGame(toroidal=Toroidal, width=N, height=N, grid_size=1)

    # Start timing the process
    start_time = time.time()

    # Create the test set
    test_data = []
    order_params = [0, 0.25, 0.5, 0.75, 1]
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

    # Save the test data to CSV
    columns = [f'State {i + 1}' for i in range(I)]
    df = pd.DataFrame(test_data, columns=columns)
    print(df.shape)
    
    save_path = os.path.join(save_folder, f'conway_test_states_{N}by{N}_{timestamp}.csv')
    df.to_csv(save_path, index=False)
    print(f"Test set saved to: {save_path}")

    # End timing the process
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
    
    # Save the animation as a GIF file
    writer = PillowWriter(fps=5)
    ani.save(filename, writer=writer)
    plt.close()

# Example usage
generate_test_sets(N=32, I=10, Toroidal=True, save_folder='patterns')