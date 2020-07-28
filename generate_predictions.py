from dataset_preparation import load_pickle, save_pickle
import numpy as np
from model_save_train import models


def preprocess_test_data(path):
    """
    :param path: Path to pickled test_set
    :return: X: reduced density matrices, W: Disorder strength that was used for generating the sample
    """
    print("Accessing ", path)
    data = load_pickle(path)
    X = [item[0] for item in data]
    # print("Input shape (Ws, Imagedim1, Imagedim2): ", np.shape(X))
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    W = np.reshape(np.asarray([item[1] for item in data]), (np.shape(data)[0], 1))
    return X, W


def load_model(path):
    return models.load_model(path)


def generate_predictions(Ns, ns, Ws):
    """
    saves predictions into lanczos/avg_prediction_sets

    :param Ns: system sizes for Model and Testset
    :param ns: block sizes for Model and Testset
    :param Ws: chosen interval for fitting
    """
    for N in Ns:
        for n in ns:
            model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
            X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) + '_Testset')
            state_prediction = model.predict(X)
            state_prediction_w = [list(x) for x in zip(state_prediction, W)]
            prediction_set = []
            for W in Ws:
                average = np.mean([item[0] for item in state_prediction_w if item[1] == W])
                prediction_set.append([average, W])
            print(prediction_set)
            save_pickle("lanczos/avg_prediction_sets/N" + str(N) + "n" + str(n) + "_prediction_set",
                        prediction_set)
    pass


if __name__ == "__main__":
    Ns = [8, 9, 10, 11]
    Ws = np.arange(0., 8.0, 0.5)
    ns = np.arange(1, 6+1, 1)
    print(ns)
    generate_predictions(Ns, ns, Ws)
