from generate_training_set import generate_training_set
from model_save_train import train_save_model, plot_model_losses
from generate_test_set import generate_test_set
from generate_predictions import generate_predictions
from plot_wc_dependency import WcPlotter

import numpy as np

if __name__ == "__main__":
    # training set
    # Ns = [8, 9, 10, 11, 12]  # up to date: 9, 10, 11
    # n_max = 6
    # Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    # repetitions = 200
    # generate_training_set(Ns, Ws, n_max, repetitions)
    # model train
    Ns = [8, 9, 10, 11, 12]
    n_max = 6
    # train_save_model(Ns, n_max,
    #                  batch_size=70,
    #                  epochs=400)
    plot_model_losses(Ns, n_max)
    # test set
    Ns = [8, 9, 10, 11, 12]
    Ws = np.arange(0., 8.0, 0.5)
    repetitions = 30
    n_max = 6
    generate_test_set(Ns, Ws, n_max, repetitions)
    # predict
    Ns = [8, 9, 10, 11, 12]
    Ws = np.arange(0., 8.0, 0.5)
    ns = np.arange(1, 6 + 1, 1)
    generate_predictions(Ns, ns, Ws)
    # get wc
    Ns = [8, 9, 10, 11, 12]
    Ws = np.arange(0., 8.0, 0.5)
    ns = np.arange(1, 6 + 1, 1)
    wc_plotter = WcPlotter(Ns, ns, Ws)
    wc_plotter.plot_all()

    #N=12 completed after 57028.43964624405 seconds.
