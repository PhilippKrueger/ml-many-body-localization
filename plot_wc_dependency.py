from dataset_preparation import load_pickle
from model_save_train import *
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.optimize import curve_fit


class WcPlotter:

    def __init__(self, Ns, ns, Ws):
        self.Ws = Ws
        self.Ns = Ns
        self.ns = ns
        self.predictions = self.get_prediction()

    def plot_all(self):
        for N in self.Ns:
            print(N)
            array = self.predictions[N - min(self.Ns), :]
            wcs = self.get_wc(array)
            title = str("Predicted phases and critical disorder strength $W_c$ "
                        + "\n over block size $n$ at system size $N=$" + str(N))
            self.plot_heat_map(wcs, array, "Disorder strength $W$", "Block size n", np.asarray(self.ns), title)
            plt.savefig('results/Wc/N' + str(N) + '_Wc_n_dependency.pdf')
            plt.close()
        for n in self.ns:
            print(n)
            array = self.predictions[:, n - min(self.ns)]
            wcs = self.get_wc(array)
            title = str("Predicted phases and critical disorder strength $W_c$ "
                        + "\n over system size $N$ at block size $n=$" + str(n))
            self.plot_heat_map(wcs, array, "Disorder strength $W$", "System size N", np.asarray(self.Ns), title)
            plt.savefig('results/Wc/n' + str(n) + '_Wc_N_dependency.pdf')
            plt.close()
        pass

    def get_wc(self, array):
        """
        Returns Ws of given array of predicted phases over system or block sizes over Ws
        """
        wcs = []
        for N_n in range(np.shape(array)[0]):
            # wcs.append(curve_fit(logistic, array[:,N_n-1], array[:,N_n-1])[0])
            nearest = np.argmin(np.abs(array[N_n - 1, :] - 0.5))#.argmin()
            print("selection:",np.abs(array[N_n - 1, :] - 0.5),"selected the element:", nearest)
            print("plotted:",array[N_n - 1, :])
            wcs.append(nearest)
        return np.asarray(wcs)

    def get_prediction(self):
        """
        Returns all predictions as N x n array of Ws
        """
        all_predictions = np.zeros((len(self.Ns), len(self.ns), len(self.Ws)))
        for N in self.Ns:
            for n in self.ns:
                element = np.array(
                    load_pickle("lanczos/avg_prediction_sets/N" + str(N) + "n" + str(n) + "_prediction_set"))
                for i in range(len(self.Ws)):
                    all_predictions[N - min(self.Ns) - 1, n - 1, i] = float(element[:, 0][i])
        return all_predictions

    def plot_heat_map(self, wcs, array, xlabel, ylabel, yticks, title):
        fig, ax = plt.subplots()
        plt.title(title)
        plt.text(0.5, len(yticks) / 2 - 0.5, 'extended', {'color': 'w', 'fontsize': 12},
                 horizontalalignment='left',
                 verticalalignment='center',
                 rotation=90,
                 )
        plt.text(3 * len(self.Ws) / 4, len(yticks) / 2 - 0.5, 'localized', {'color': 'k', 'fontsize': 12},
                 horizontalalignment='left',
                 verticalalignment='center',
                 rotation=90,
                 )
        norm = col.Normalize(vmin=0, vmax=1)
        pos = ax.imshow(array, cmap='bwr', vmin=0.0, vmax=1.0, norm=norm)  # aspect=1, #Purples
        ax.scatter(wcs, yticks - min(yticks), s=100, c="w", marker='^', label='$W_c$', edgecolors="k") #wcs
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pos, cax=cax)
        # ticks
        ax.set_xticks(np.arange(len(self.Ws)))
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_xticklabels(self.Ws)
        ax.set_yticklabels(yticks)
        ax.legend()
        plt.tight_layout()
        pass

    def find_intersection(self, array):
        p = np.polyfit(self.Ws, array, 1)
        return (0.5 - p[0])/p[1]


def logistic(x, a):
    return 1 / (1 + np.exp(-50 * (x - a)))


def heaviside(x, a):
    return 0.5 * np.sign(x - a) + 0.5


def linear(x, a):
    return a * x


def load_model(path):
    return models.load_model(path)


if __name__ == "__main__":
    Ns = [8, 9, 10, 11, 12]
    Ws = np.arange(0., 8.0, 0.5)
    ns = np.arange(1, 6 + 1, 1)
    wc_plotter = WcPlotter(Ns, ns, Ws)
    wc_plotter.plot_all()
