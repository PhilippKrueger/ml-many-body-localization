from dataset_preparation import load_pickle
from model_save_train import *
from scipy.optimize import curve_fit


class WcPlotter:

    def __init__(self, Ns, ns, Ws):
        self.Ws = Ws
        self.Ns = Ns
        self.ns = ns
        self.predictions = self.get_prediction()

    def plot_all(self):
        for N in self.Ns:
            array = self.predictions[N-min(self.Ns), :]
            print(array)
            wcs = self.get_wc(array)
            title = str("Predicted phases and critical disorder strength $W_c$ "
            + "\n over block size $n$ at system size $N=$" + str(N))
            self.plot_heat_map(wcs, array, "Disorder strength $W$", "Block size n", title)
            plt.savefig('results/Wc/N' + str(N) + '_Wc_n_dependency.pdf')
            plt.close()
        for n in self.ns:
            array = self.predictions[:, n]
            print(array)
            wcs = self.get_wc(array)
            title = str("Predicted phases and critical disorder strength $W_c$ "
            + "\n over system size $N$ at block size $n=$" + str(n))
            self.plot_heat_map(wcs, array, "Disorder strength $W$", "System size N", title)
            plt.savefig('results/Wc/n' + str(n) + '_Wc_N_dependency.pdf')
            plt.close()
        pass

    def get_wc(self, array):
        """
        Returns Ws of given array of predicted phases over system or block sizes over Ws
        """
        wcs = []
        for N_n in np.shape(array):
            wcs.append(curve_fit(logistic, array[N_n][:, 0], array[N_n][:, 1])[0])
        return wcs

    def get_prediction(self):
        """
        Returns all predictions as N x n array of Ws
        """
        all_predictions = np.zeros((len(self.Ns), len(self.ns)))
        for N in self.Ns:
            for n in self.ns:
                all_predictions[N+min(self.Ns), n] = load_pickle("lanczos/avg_prediction_sets/N" + str(N) + "n" + str(n) + "_prediction_set") #
        return all_predictions

    def plot_heat_map(self, wcs, array, xlabel, ylabel, title):
        fig, ax = plt.subplots()
        plt.title(title)
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
        pos = ax.imshow(array, extent='auto', aspect=0.5, cmap='bwr')
        ax.scatter(wcs[:, 0], wcs[:, 1] - 0.5, s=100, c="w", marker='^', label='$W_c$', edgecolors="k")
        fig.colorbar(pos, ax=ax)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        ax.legend()
        plt.tight_layout()
        pass


def logistic(x, a):
    return 1 / (1 + np.exp(-50 * (x - a)))


def heaviside(x, a):
    return 0.5 * np.sign(x - a) + 0.5


def load_model(path):
    return models.load_model(path)


if __name__ == "__main__":
    Ns = [8]
    Ws = np.arange(0., 8.0, 0.5)
    ns = np.arange(1, 6, 1)
    wc_plotter = WcPlotter(Ns, ns, Ws)
    wc_plotter.plot_all()