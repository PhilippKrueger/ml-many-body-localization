from model_save_train import *
from scipy.optimize import curve_fit


def preprocess_test_data(path):
    """
    :param path: Path to pickled test_set
    :return: X: reduced density matrices, W: Disorder strength that was used for generating the sample
    """
    print("Accessing ",path)
    data = load_pickle(path)
    X = [item[0] for item in data]
    # print("Input shape (Ws, Imagedim1, Imagedim2): ", np.shape(X))
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    W = np.reshape(np.asarray([item[1] for item in data]), (np.shape(data)[0], 1))
    return X, W


def logistic(x, a):
    return 1 / (1 + np.exp(-50 * (x - a)))


def heaviside(x, a):
    return 0.5*np.sign(x-a)+0.5


def load_model(path):
    return models.load_model(path)


def get_wc(N, n, Ws, repetitions):
    """
    Calculates Wc

    :param N: system size for Model and Testset
    :param n: block size for Model and Testset
    :param Ws: chosen interval for fitting
    :param repetitions: Number of datapoints per W_pred
    :return: Wc, n
    """
    model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
    X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) +'_Testset')

    state_prediction = model.predict(X)
    state_prediction = np.reshape(state_prediction, (int(len(state_prediction)/repetitions), repetitions))
    state_prediction = np.mean(state_prediction, axis=1)

    popt, pcov = curve_fit(logistic, Ws, np.reshape(state_prediction, (len(state_prediction))))  # state_prediction.astype(np.float))
    # plot_wc_fit(N,popt,state_prediction)
    return popt[0]# , n #, N #, np.shape(X[0])[0]

def get_wc_N(N, n, Ws, repetitions):
    """
    Calculates Wc

    :param N: system size for Model and Testset
    :param n: block size for Model and Testset
    :param Ws: chosen interval for fitting
    :param repetitions: Number of datapoints per W_pred
    :return: Wc, n
    """
    model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
    X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) +'_Testset')

    state_prediction = model.predict(X)
    state_prediction = np.reshape(state_prediction, (int(len(state_prediction)/repetitions), repetitions))
    state_prediction = np.mean(state_prediction, axis=1)

    popt, pcov = curve_fit(logistic, Ws, np.reshape(state_prediction, (len(state_prediction))))  # state_prediction.astype(np.float))
    # plot_wc_fit(N,popt,state_prediction)
    return popt[0]# , N #, N #, np.shape(X[0])[0]


class HeatMapPlotter:

    def __init__(self, Ns, Ws, n_max, repetitions):
        self.Ns = Ns
        self.Ws = Ws
        self.n_max = n_max
        self.repetitions = repetitions


    def predict_w_n(self):
        W_preds = {system_size : [] for system_size in self.Ns}
        for N in self.Ns:
            for n in range(1, self.n_max + 1):
                model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
                X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) + '_Testset')
                W_preds[N].append(model.predict(X))
        return W_preds

    def fit_wc_n(self):
        W_c_fit = {system_size : [] for system_size in self.Ns}
        for N in self.Ns:
            for n in range(1, self.n_max + 1):
                W_c_fit[N].append((get_wc(N, n, self.Ws, self.repetitions), n))
        return W_c_fit

    def predict_w_N(self):
        W_preds = {block_size : [] for block_size in range(1, self.n_max+1)}
        for n in range(1, self.n_max + 1):
            for N in self.Ns:
                model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
                X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) + '_Testset')
                W_preds[n].append(model.predict(X))
        return W_preds

    def fit_wc_N(self):
        W_c_fit = {block_size : [] for block_size in range(1, self.n_max+1)}
        for n in range(1, self.n_max + 1):
            for N in self.Ns:
                W_c_fit[n].append((get_wc_N(N, n, self.Ws, self.repetitions),N))
        return W_c_fit

    def plot_wc_heatmap_n(self):
        """
        Plots Heatmap with blocksize and W_pred

        W_pred: W x n array
        W_c_fit: W_c(n) x 1 array
        """
        self.W_preds = self.predict_w_n()
        self.W_c_fit = self.fit_wc_n()

        for N in self.Ns:
            W_pred = np.asarray(self.W_preds[N])
            W_pred = np.reshape(W_pred, (np.shape(W_pred)[0],np.shape(W_pred)[1]))
            W_c_fit = np.array(self.W_c_fit[N])


            # W_c_fit = np.reshape(W_c_fit, (np.shape(W_c_fit)[0], np.shape(W_c_fit)[1]))
            fig, ax = plt.subplots()
            plt.title("Predicted phases and critical disorder strength $W_c$ \n over block size $n$ at system size $N=$" + str(N))
            plt.text(0.5, 3.5, 'extended', {'color': 'w', 'fontsize': 12},
                     horizontalalignment='left',
                     verticalalignment='center',
                     rotation=90,
                     )
            plt.text(3.5, 3.5, 'localized', {'color': 'k', 'fontsize': 12},
                     horizontalalignment='left',
                     verticalalignment='center',
                     rotation=90,
                     )
            pos = ax.imshow(W_pred, extent=(0, 4, 0, 7), aspect=0.5, cmap='bwr')
            fig.colorbar(pos, ax=ax)
            ax.scatter(W_c_fit[:,0], W_c_fit[:,1]-0.5, s=100, c="w", marker='^', label='$W_c$', edgecolors="k")
            plt.ylabel("Block size n")
            plt.xlabel("Predicted disorder strength $W_{pred}$")
            ax.legend()
            plt.tight_layout()
            plt.savefig('results/Wc/N'+str(N)+'_Wc_n_dependency.pdf')
            plt.close()
        pass

    def plot_wc_heatmap_N(self):
        """
        Plots Heatmap with blocksize and W_pred

        W_pred: W x n array
        W_c_fit: W_c(n) x 1 array
        """
        self.W_preds = self.predict_w_N()
        self.W_c_fit = self.fit_wc_N()

        for n in range(1, self.n_max+1):
            W_pred = np.asarray(self.W_preds[n])
            W_pred = np.reshape(W_pred, (np.shape(W_pred)[0],np.shape(W_pred)[1]))
            W_c_fit = np.array(self.W_c_fit[n])


            # W_c_fit = np.reshape(W_c_fit, (np.shape(W_c_fit)[0], np.shape(W_c_fit)[1]))
            fig, ax = plt.subplots()
            plt.title("Predicted phases and critical disorder strength $W_c$ \n over system size $N$ with block size $n=$" + str(n))
            plt.text(0.5, 3.5, 'extended', {'color': 'w', 'fontsize': 12},
                     horizontalalignment='left',
                     verticalalignment='center',
                     rotation=90,
                     )
            plt.text(3.5, 3.5, 'localized', {'color': 'k', 'fontsize': 12},
                     horizontalalignment='left',
                     verticalalignment='center',
                     rotation=90,
                     )
            pos = ax.imshow(W_pred, extent=(0, 4, self.Ns[0]-1, self.Ns[-1]), aspect='auto', cmap='bwr')
            # Shift ticks to be at 0.5, 1.5, etc
            # ax.yaxis.set(ticks=np.arange(0.5, len(self.Ns)), ticklabels=map(str, input(self.Ns)))

            fig.colorbar(pos, ax=ax)
            # print(W_c_fit)
            ax.scatter(W_c_fit[:,0], W_c_fit[:,1]-0.5, s=100, c="w", marker='^', label='$W_c$', edgecolors="k")
            plt.ylabel("System size L")
            plt.xlabel("Predicted disorder strength $W_{pred}$")
            ax.legend()
            plt.tight_layout()
            plt.savefig('results/Wc/n'+str(n)+'_Wc_N_dependency.pdf')
            plt.close()
        pass

    def plot_wc_fit(self, N, popt, state_prediction):
        fig, ax1 = plt.subplots()
        ax1 = plt.scatter(self.Ws, state_prediction)
        ax1 = plt.plot(self.Ws, logistic(self.Ws, *popt), 'k')

        plt.title('Phase prediction $N = $' + str(N) + ", $W_c = $" + "{0:.3g}".format(popt[0]))
        plt.ylabel('Probability of localized phase')
        plt.xlabel('$W_{max}$')
        plt.legend(['Logistic fit', 'Predicted phase'], loc='upper left')
        plt.savefig('results/N' + str(N) + '_predict_wc.pdf')
        pass


if __name__ == "__main__":
    Ns = [9, 10]
    Ws = np.arange(0., 4.0, 0.05)
    n_max = 7
    repetitions = 5
    heat_map_plotter = HeatMapPlotter(Ns, Ws, n_max, repetitions)
    heat_map_plotter.plot_wc_heatmap_n()
    print("done")
    heat_map_plotter.plot_wc_heatmap_N()
    print("done")
