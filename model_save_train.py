from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import layers, models, losses, callbacks
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k
import time


def load_pickle(filename, to_numeric=1):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_training_data(path): # reduced_rho, W, self.N, n, E
    data = load_pickle(path)
    X = data
    X = [item[0] for item in X]
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    y = data
    y = np.reshape(np.asarray([map_target(item[1]) for item in data]), (np.shape(y)[0], 1))
    return X, y

def map_target(item):
    if item == 0.5:
        return 0 # ergodic/delocalized phase
    elif item == 8.0:
        return 1 # localized phase
    else:
        print("Invalid training data.")

def mean_pred(y_true, y_pred):
    return k.mean(y_pred)


class ModelTrainer:

    def __init__(self, x, y, N, n_max):
        self.N = N
        self.n_max = n_max
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.model = self.generate_model_sparse()

    def train_test_split(self):
        pass

    def generate_model(self):
        model = models.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='mae', metrics=['accuracy'])#loss used to be mae loss # metrics: 'mean_absolute_error', 'mean_squared_error',
        return model

    def generate_model_sparse(self):
        model = models.Sequential()
        # if self.N != 12:
        # model.add(layers.Conv2D(32, (6, 6), activation='relu', input_shape=(np.shape(self.X_train)[1], np.shape(self.X_train)[1], 2)))
        # model.add(layers.MaxPooling2D((4, 4)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer='l2'))
        model.add(layers.Dense(64, activation='relu', bias_regularizer='l2'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])#loss used to be mae loss # metrics: 'mean_absolute_error', 'mean_squared_error',
        return model

    def score(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('test loss, test acc:', score)
        pass

    def fit_model(self, batch_size, epochs):
        history = self.model.fit(self.X_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2,
                       validation_data=(self.X_test, self.y_test)
                       )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)

    def training_history(self, history):
        print(history.history.keys())
        #  "Accuracy"
        fig, ax1 = plt.subplots()
        ax1 = plt.plot(history.history['acc'])
        ax1 = plt.plot(history.history['val_acc'])
        ax1 = plt.title('Model accuracy and loss')
        # "Loss"
        ax1 = plt.plot(history.history['loss'])
        ax1 = plt.plot(history.history['val_loss'])
        ax1 = plt.xlabel('Epoch')
        ax1 = plt.legend(['Training set accuracy', 'Validation set accuracy','Training set loss', 'Validation set loss']
                         , loc='center right')
        plt.savefig("results/accuracy_loss_epochs/N"+str(self.N)+"n"+str(self.n_max)+"_accuracy_loss_epochs.pdf")
        pass

def train_save_model(Ns, n_max, batch_size, epochs):
    start_time = time.time()
    for N in Ns:
        start_model_time = time.time()
        for n in range(1, n_max+1):
            X, y = preprocess_training_data("lanczos/training_sets/N"+str(N)+"n"+str(n)+"_Trainset")
            model_trainer = ModelTrainer(X, y, N, n_max)
            history = model_trainer.fit_model(batch_size=batch_size,
                                              epochs=epochs)
            model_trainer.training_history(history)
            model_trainer.save_model("lanczos/models/N"+str(N)+"n"+str(n)+"_Model")
        print("--- Model trainings for N=" + str(N) + " lasted %s seconds ---" % (
                        time.time() - start_model_time))
    print("--- Model training lasted %s seconds ---" % (time.time() - start_time))
    pass


if __name__ == "__main__":
    # Ns = [10, 11, 12]
    Ns = [11, 12]
    n_max = 7
    train_save_model(Ns, n_max,
                     batch_size=70,
                     epochs=40)

    # N = 12 Model training lasted 537.23 seconds
