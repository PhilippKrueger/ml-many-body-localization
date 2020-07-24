import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import pickle


def load_pickle(filename, to_numeric=1):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def preprocess_training_data(path):  # reduced_rho, W, self.N, n, E
    data = load_pickle(path)
    X = data
    X = [item[0] for item in X]
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    y = data
    y = np.reshape(np.asarray([map_target(item[1]) for item in data]), (np.shape(y)[0], 1))
    print("Number of samples:", len(X))
    return X, y


def map_target(item):
    if item == 0.5:
        return 0  # ergodic/delocalized phase
    elif item == 8.0:
        return 1  # localized phase
    else:
        print("Invalid training data.")


def save_ground_state_figures(Ns, n_max):
    for N in Ns:
        for n in range(1, n_max+1):
            data_list = load_pickle("lanczos/training_sets/N" + str(N) + "n" + str(n) + "_Trainset")
            try:
                save_groundstate_figure(get_ergodic(data_list))
                save_groundstate_figure(get_localized(data_list))
            except:
                print("Ground State Figure for N=" + str(N) + ",n=" + str(n) + " could not be generated")
    pass


def get_ergodic(training_set):
    ergodic = [item for item in training_set if item[1] == 0.5]  # len: repetitions
    ergodic = sorted(ergodic, key=itemgetter(4))[0]  # sort by lowest E
    return ergodic


def get_localized(training_set):
    localized = [item for item in training_set if item[1] == 8.0]  # len: repetitions
    localized = sorted(localized, key=itemgetter(4))[0]  # sort by lowest E
    return localized


def save_groundstate_figure(sample):  # reduced_rho, W, self.N, n, E, rep
    """
    Plots a heatmap to the lowest groundstate of a specified system and block size.

    :param N: system size
    :param training_set: tra
    :param n: block size
    :return:
    """
    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(sample[0]), cmap='Purples')
    fig.colorbar(pos, ax=ax1)
    plt.title("Reduced density matrix for $n=$" + str(sample[3]) + " consecutive sites \n at $E=$"
              + str(round(sample[4], 2)) + " for $W=$" + str(sample[1]) + ", $N = $" + str(sample[2]))
    plt.savefig(
        "results/groundstates/N" + str(sample[2]) + "n" + str(sample[3]) + "_trainingset_groundstate_Wmax" + str(
            sample[1]) + ".pdf")
    plt.close()
    pass


if __name__ == "__main__":
    Ns = [8, 9, 10, 11]
    n_max = 6
    save_ground_state_figures(Ns, n_max)
