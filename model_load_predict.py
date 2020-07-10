from tensorflow.keras import layers, models
from generate_training_set import TrainingSetGenerator, save_pickle
import numpy as np
from model_save_train import *
from scipy.optimize import curve_fit

def load_model(path):
    return models.load_model(path)

def preprocess_test_data(path):
    data = load_pickle(path)
    X = [item[0] for item in data]
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    W = np.reshape(np.asarray([item[1] for item in data]), (np.shape(data)[0], 1))
    return X, W

def logistic(x, a, b):
    return 1/(1+np.exp(-a*(x-b)))

def get_wc(N):
    model = load_model('pickled/'+N+'_Model')
    X, W = preprocess_test_data('pickled/'+N+'_Testset')
    state_prediction = model.predict(X)
    plt.scatter(W_max, state_prediction)

    print(W_max, np.shape(W_max))

    popt, pcov = curve_fit(logistic, np.asarray(W_max), np.asarray(state_prediction))
    plt.scatter(W_max, logistic(W_max, *popt))

    plt.title('Phase prediction')
    plt.ylabel('Probability of localized phase')
    plt.xlabel('W')
    plt.legend(['Prediction', 'Logistic Fit'], loc='upper left')
    plt.savefig('results/'+N+'_predict_wc.jpg')


if __name__ == "__main__":
    Ns = [10]
    W_max = np.arange(0., 4.0, 0.05)
    training_set_generator = TrainingSetGenerator(Ns, W_max)
    save_pickle("pickled/N10_Testset", training_set_generator.training_set)

    get_wc('N10')

