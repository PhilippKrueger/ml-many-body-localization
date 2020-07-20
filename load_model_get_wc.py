from generate_training_set import TrainingSetGenerator, save_pickle
from model_save_train import *
from scipy.optimize import curve_fit


def preprocess_test_data(path):
    print(path)
    data = load_pickle(path)
    X = [item[0] for item in data]
    print("Input shape (Ws, Imagedim1, Imagedim2): ", np.shape(X))
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


def get_wc(N, n, W_max):
    model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
    X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) +'_Testset')
    state_prediction = model.predict(X)


    # print(W_max, np.shape(W_max))
    popt, pcov = curve_fit(logistic, W_max, np.reshape(state_prediction, (len(state_prediction))))  # state_prediction.astype(np.float))
    # plot_wc_fit(N,popt,state_prediction)
    return popt[0], n #, N, np.shape(X[0])[0]




def plot_wc_dependencies(Ns, W_max):
    Wc_dependencies = []
    for N in Ns:
        for n in range(1, N+1):
            Wc_dependencies.append(get_wc(N, n, W_max))
    Wc_dependencies = np.array([np.array(xi) for xi in Wc_dependencies])
    print(Wc_dependencies)
    fig, ax1 = plt.subplots()
    plt.title("$W_c$ dependency on system size L")
    plt.ylabel('Critical maximum disorder strength $W_c$')
    plt.xlabel('System size $L$')
    ax1 = plt.scatter(Wc_dependencies[:, 1], Wc_dependencies[:, 0])
    plt.savefig('results/Wc_L_dependency.pdf')
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("$W_c$ dependency on block size n")
    plt.ylabel('Critical maximum disorder strength $W_c$')
    plt.xlabel('Block size $n$')
    ax1 = plt.scatter(Wc_dependencies[:, 2], Wc_dependencies[:, 0])
    plt.savefig('results/Wc_n_dependency.pdf')
    plt.close()
    pass

class HeatMapPlotter:

    def __init__(self, Ns, Ws, n_max):
        self.Ns = Ns
        self.Ws = Ws
        self.n_max = n_max
        self.W_preds = self.predict_w()
        self.W_c_fit = self.fit_wc()

    def predict_w(self):
        W_preds = {system_size : [] for system_size in self.Ns}
        for N in self.Ns:
            for n in range(1, self.n_max + 1):
                model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
                X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) + '_Testset')
                W_preds[N].append(model.predict(X))
        return W_preds

    def fit_wc(self):
        W_c_fit = {system_size : [] for system_size in self.Ns}
        for N in self.Ns:
            for n in range(1, self.n_max + 1):
                W_c_fit[N].append(get_wc(N, n, self.Ws))
        return W_c_fit

    def plot_wc_heatmap(self):
        """
        W_pred: W x n array
        W_c_fit: W_c(n) x 1 array
        :return:
        """
        for N in self.Ns:
            W_pred = np.asarray(self.W_preds[N])
            W_pred = np.reshape(W_pred, (np.shape(W_pred)[0],np.shape(W_pred)[1]))
            W_c_fit = np.array(self.W_c_fit[N])

            # print(W_c_fit)
            # print(np.shape(W_c_fit))

            # W_c_fit = np.reshape(W_c_fit, (np.shape(W_c_fit)[0], np.shape(W_c_fit)[1]))
            fig, ax = plt.subplots()
            plt.title("Predicted phases and $W_c$ over block size $n$, $N=$" + str(N))
            pos = ax.imshow(W_pred, extent=(0, 4, 0, 7), aspect=0.5, cmap='bwr')
            fig.colorbar(pos, ax=ax)
            ax.scatter(W_c_fit[:,0], W_c_fit[:,1]-0.5, s=100, c="w", marker='^', label='$W_c$', edgecolors="k")
            plt.ylabel("Block size n")
            plt.xlabel("Predicted disorder strength $W_{predicted}$")
            ax.legend()
            plt.savefig('results/Wc/N'+str(N)+'_Wc_n_dependency.pdf')
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

    # fixme add n plots over N


if __name__ == "__main__":
    Ns = [11, 12]
    Ws = np.arange(0., 4.0, 0.05)
    n_max = 7
    # plot_wc_dependencies(Ns, Ws)
    heat_map_plotter = HeatMapPlotter(Ns, Ws, n_max)
    heat_map_plotter.plot_wc_heatmap()
