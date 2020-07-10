from tensorflow.keras import layers, models
from generate_training_set import TrainingSetGenerator, save_pickle
import numpy as np
from model_save_train import *

def load_model(path):
    return models.load_model(path)

def preprocess_test_data(path):
    data = load_pickle(path)
    X = [item[0] for item in data]
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    W = np.reshape(np.asarray([item[1] for item in data]), (np.shape(data)[0], 1))
    return X, W

if __name__ == "__main__":
    Ns = [10]
    W_max = np.arange(0.5, 8.0, 0.1)
    training_set_generator = TrainingSetGenerator(Ns, W_max)
    save_pickle("pickled/N10_Testset", training_set_generator.training_set)

    model = load_model('pickled/N10_Model')
    X, W = preprocess_test_data("pickled/N10_Testset")
    state_prediction = model.predict(X)

    plt.title('Phase prediction')
    plt.ylabel('P(localized)')
    plt.xlabel('W')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.scatter(W_max, state_prediction)
    plt.show()