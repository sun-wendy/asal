import sys
import os
import pandas as pd
import time
from datetime import datetime #make sure this notebook is in the same directory as the conway_lib folder
from conway_lib import ConwayGame
conway_game=ConwayGame()


def generate_sets(A=100, N=30, I=2, Toroidal=False, save_folder='Conway_GPT'):
    # Create a timestamp for the current date and time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Ensure that the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Create an instance of ConwayGame
    game = ConwayGame(width=N, height=N, grid_size=1)

    # Start timing the process
    start_time = time.time()
    
    # Generate the validation sets using the method provided
    start = 0 #order param of the frist sample
    end = 1 #order param of the last sample
    train_data = game.generate_sets(A=A, N=N, I=I, s=start, e=end)

    # Process and save the data to CSV
    data = []
    for index in range(A):
        states = train_data[index]
        data.append(states)
        
    # Save the flattened states to CSV
    columns = [f'State {i + 1}' for i in range(I)]
    df = pd.DataFrame(data, columns=columns)
    print(df.shape)
    
    toroidal_str = 'toroidal' if Toroidal else 'non_toroidal'
    save_path = os.path.join(save_folder, f'conway_states_{start}_{end}_{A}by{N}by{N}by{I}_{toroidal_str}_{timestamp}.csv')
    df.to_csv(save_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"CSV generation time: {elapsed_time} seconds")


if __name__ == "__main__":
    generate_sets(A=100, N=64, I=256, Toroidal=True, save_folder='Conway_GPT')
