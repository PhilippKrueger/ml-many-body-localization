from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import time
from dataset_preparation import preprocess_training_data
from tqdm import trange
from numpy import genfromtxt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ModelTrainer:

    def __init__(self, x, y, N, n):
        self.N = N
        self.n = n
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        self.model = self.generate_model_sparse()

    def generate_model_sparse(self):
        model = models.Sequential()
        if self.n > 3:
            filters = self.n*self.n
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(np.shape(self.X_train[0])[0], np.shape(self.X_train[0])[1], 2)))
            model.add(layers.MaxPooling2D((4, 4)))
            model.add(layers.Flatten())
        else:
            model.add(layers.Flatten(input_shape=(np.shape(self.X_train)[1], np.shape(self.X_train)[1], 2)))
            model.add(layers.Dense(32, activation='relu')),

        model.add(layers.Dropout(rate=0.3)) # fixme not tested yet
        model.add(layers.Dense(64, activation='relu', bias_regularizer='l2'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # model.summary()
        return model

    def score(self):
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("test loss:"+"{:.3E}".format(score[0])+", test acc:" + "{:.0%}".format(score[1]))
        pass

    def fit_model(self, batch_size, epochs):
        csv_logger = callbacks.CSVLogger("lanczos/models/N"+str(self.N)+"n"+str(self.n)+"_model_loss.csv",
                                         separator=",",
                                         append=False)
        history = self.model.fit(self.X_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=0,#2
                       validation_data=(self.X_test, self.y_test),
                       callbacks=[csv_logger]
                       )
        return history

    def save_model(self, filepath):
        self.model.save(filepath)

    def training_history(self, history, n, N):

        fig, ax1 = plt.subplots()
        plt.title('Model accuracy and loss for $n=$'+str(n)+', $N=$'+str(N))
        plt.xlabel('Training epoch')

        # "Loss"
        ax1.set_ylabel('Accuracy')  # we already handled the x-label with ax1
        ax1.tick_params(axis='y')
        ln1 = ax1.plot(history.history['acc'], 'r', label='Training set accuracy')
        ln2 = ax1.plot(history.history['val_acc'], 'g', label='Validation set accuracy')


        #  "Accuracy"
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Logarithmic loss')  # we already handled the x-label with ax1
        ax2.set_yscale('log')
        ax2.tick_params(axis='y')
        ln3 = ax2.plot(history.history['loss'], label='Training set loss')
        ln4 = ax2.plot(history.history['val_loss'], label='Validation set loss')


        # Joined Legend
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.tight_layout()
        plt.savefig("results/accuracy_loss_epochs/N"+str(self.N)+"n"+str(n)+"_accuracy_loss_epochs.pdf")
        print("Scores for N=" + str(N) + ", n=" + str(n))
        plt.close()
        self.score()
        pass

def train_save_model(Ns, n_max, batch_size, epochs):
    start_time = time.time()
    for N in Ns:
        start_model_time = time.time()
        for n in trange(1, n_max+1):
            X, y = preprocess_training_data(str("lanczos/training_sets/N"+str(N)+"n"+str(n)+"_Trainset"))
            model_trainer = ModelTrainer(X, y, N, n)
            history = model_trainer.fit_model(batch_size=batch_size,
                                              epochs=epochs)
            model_trainer.training_history(history, n, N)
            model_trainer.save_model("lanczos/models/N"+str(N)+"n"+str(n)+"_Model")
        print("--- Model trainings for N=" + str(N) + " lasted %s seconds ---" % (
                        time.time() - start_model_time))
    print("--- Model training lasted %s seconds ---" % (time.time() - start_time))
    pass

def get_metric(metric, Ns, n_max):
    """
    :param metric: 0:epoch, 1:acc, 2:loss, 3:val_acc, 4:val_loss
    :return: metric values per system and block size
    """
    values = np.zeros((len(Ns), n_max))
    for N in range(0, len(Ns)):
        for n in range(0, n_max):
            path = "lanczos/models/N" + str(min(Ns) + N) + "n" + str(n + 1) + "_model_loss.csv"
            my_data = genfromtxt(path, delimiter=',')
            values[N, n] = float(my_data[-1, metric])  # val loss 4
    return values

def plot_model_losses(Ns, n_max):
    titles = ["Epochs", "Training accuracy", "Training losses", "Validation accuracy", "Validation losses"]
    for train_val in [1, 2, 3, 4]:
        losses = get_metric(train_val, Ns, n_max)
        ns = np.arange(1, n_max+1, 1)
        fig, ax = plt.subplots()
        im = ax.imshow(losses, cmap='Purples')
        ax.set_xticks(np.arange(len(ns)))
        ax.set_yticks(np.arange(len(Ns)))
        ax.set_xticklabels(ns)
        ax.set_yticklabels(Ns)
        for i in range(len(Ns)):
            for j in range(len(ns)):
                if train_val == 1 or train_val == 3:
                    text = ax.text(j, i, "{:.0%}".format(losses[i, j]),
                                   ha="center", va="center", color="k")
                else:
                    text = ax.text(j, i, "{0:.2E}".format(losses[i, j]),
                               ha="center", va="center", color="k")

        ax.set_title(titles[train_val])
        plt.xlabel("Block size")
        plt.ylabel("System size")
        # colorbar matches figure height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        fig.tight_layout()
        plt.savefig("results/accuracy_loss_epochs/all_"+titles[train_val].lower().replace(' ', '_')+".pdf")
    pass


if __name__ == "__main__":
    Ns = [8, 9, 10, 11]
    n_max = 6
    # train_save_model(Ns, n_max,
    #                  batch_size=70,
    #                  epochs=100)
    plot_model_losses(Ns, n_max, set)
