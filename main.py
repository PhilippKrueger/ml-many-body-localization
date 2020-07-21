from generate_test_set import generate_test_set
from generate_training_set import generate_training_set
from load_model_get_wc import *
from model_save_train import train_save_model
import numpy as np

if __name__ == "__main__":
    # Training set generation
    Ns = [8]
    n_max = 7
    Ws=[0.5, 8.0]
    repetitions = 300

    generate_training_set(Ns, Ws, n_max, repetitions)

    # Model training
    train_save_model(Ns, n_max,
                     batch_size=70,
                     epochs=200) #40

    # Test set generation
    repetitions = 50
    Ws = np.arange(0., 4.0, 0.5)  # => ergodic/delocalized phase, 8.0 localized phase
    sample_size = generate_test_set(Ns, Ws, n_max, repetitions)

    # W_c dependency
    repetitions = 5*6
    heat_map_plotter = HeatMapPlotter(Ns, Ws, n_max, repetitions)
    heat_map_plotter.plot_wc_heatmap_n()
    heat_map_plotter.plot_wc_heatmap_N()
