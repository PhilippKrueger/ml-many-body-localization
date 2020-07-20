from generate_test_set import generate_test_set
from generate_training_set import generate_training_set
from load_model_get_wc import *
from model_save_train import train_save_model
import numpy as np

if __name__ == "__main__":
    # Training set generation
    Ns = [11, 12]
    n_max = 7
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    repetitions = 500
    generate_training_set(Ns, Ws, n_max, repetitions)

    # Model training
    train_save_model(Ns, n_max,
                     batch_size=32,
                     epochs=40)
    # Test set generation
    W_max = np.arange(0., 4.0, 0.05)
    repetitions = 1
    generate_test_set(Ns, W_max, repetitions)
    # W_c dependency
    n_max = 7
    # plot_wc_dependencies(Ns, Ws)
    heat_map_plotter = HeatMapPlotter(Ns, Ws, n_max)
    heat_map_plotter.plot_wc_heatmap()
