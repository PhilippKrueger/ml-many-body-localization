from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import layers, models, losses, callbacks
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k


def load_pickle(filename, to_numeric=1):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess_training_data(path):
    data = load_pickle(path)
    X = data
    X = [item[0] for item in X]
    # print(np.shape(X))
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    # print("X shape:", np.shape(X))
    y = data
    y = np.reshape(np.asarray([map_target(item[1]) for item in data]), (np.shape(y)[0], 1))
    # print("y values:", y[0])
    # print("y shape:", np.shape(y))
    # for x in L:
    #     del x[3]
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

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.model = self.generate_model()

    def train_test_split(self):
        pass

    def generate_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 2)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='mae', metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])
        return model

    def score(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('test loss, test acc:', score)
        pass

    def fit_model(self, batch_size, epochs):
        tensorboard_callback = callbacks.TensorBoard(log_dir="./logs") #fixme not sure how to use this
        history = self.model.fit(self.X_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(self.X_test, self.y_test)
                       )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)

    def training_history(self, history):
        print(history.history.keys())
        #  "Accuracy"
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()


if __name__ == "__main__":
    X, y = preprocess_training_data("pickled/N10_Trainset")
    model_trainer = ModelTrainer(X, y)
    history = model_trainer.fit_model(batch_size=100, epochs=10)
    model_trainer.training_history(history)
    model_trainer.save_model("pickled/N10_Model")