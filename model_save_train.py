from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import layers, models
import numpy as np


def load_pickle(filename, to_numeric=1):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def preprocess(path):
    data = load_pickle(path)
    # for x in L:
    #     del x[3]
    return data


class ModelTrainer:

    def __init__(self, x, y):
        self.X_test, self.X_train, self.y_test, self.y_train = train_test_split(x, y, test_size=0.3, random_state=42)
        self.model = self.generate_model(x, y)

    def train_test_split(self):
        pass

    def generate_model(self, x, y):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        return model.compile()

    def score(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        pass

    def fit_model(self, batch_size, epochs, ):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test)
                       )
        pass

    def save_model(self):
        self.model.save("pickled/model")


if __name__ == "__main__":
    data = preprocess("pickled/N10")
    print(np.shape(data))
    print(np.shape(data[:][0]))
    model_trainer = ModelTrainer(data[:][0], data[:][0])
