from ed import *
import time
import pickle
# import qutip
import matplotlib.pyplot as plt
# from tqdm import trange, tqdm
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from operator import itemgetter


def generate_training_set(Ns, Ws, n_max, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, Ws, n_max, repetitions)
        print("Training Set N=" + str(N) + " completed after %s seconds." % (time.time() - start_time))
        for n in range(1, n_max + 1):
            save_groundstate_figures(N, training_set_generator.training_set[n], n)
            save_pickle("lanczos/training_sets/N" + str(N) + "n" + str(n) + "_Trainset",
                        training_set_generator.training_set[n])
    print("--- Training set generation lasted %s seconds ---" % (time.time() - start_time))
    pass


def save_groundstate_figures(N, training_set, n):  # reduced_rho, W, self.N, n, E, rep
    """
    Plots a heatmap to the lowest groundstate of a specified system and block size.

    :param N: system size
    :param training_set: tra
    :param n:
    :return:
    """
    ergodic = [item for item in training_set if item[1] == 0.5 and item[-1] == 0]  # len: repetitions
    localized = [item for item in training_set if item[1] == 8.0 and item[-1] == 0]  # len: repetitions

    ergodic = sorted(ergodic, key=itemgetter(4))[0] # sort by lowest E
    localized = sorted(localized, key=itemgetter(4))[0] # sort by lowest E

    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(ergodic[0]), cmap='bwr')
    fig.colorbar(pos, ax=ax1)
    plt.title("Reduced density matrix for $n=$" + str(n) + " consecutive sites \n at $E=$"
              + str(round(ergodic[4], 2)) + " for $W=$" + str(ergodic[1]) + ", $N = $" + str(N))
    plt.savefig(
        "results/groundstates/N" + str(N) + "n" + str(n) + "_trainingset_groundstate_Wmax" + str(ergodic[1]) + ".pdf")
    plt.close()

    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(localized[0]), cmap='bwr')
    fig.colorbar(pos, ax=ax1)
    plt.title("Reduced density matrix for $n=$" + str(localized[3]) + " consecutive sites \n at $E=$"
              + str(round(localized[4], 2)) + " for $W=$" + str(localized[1]) + ", $N = $" + str(N))
    plt.savefig(
        "results/groundstates/N" + str(N) + "n" + str(localized[3]) + "_trainingset_groundstate_Wmax" + str(
            localized[1]) + ".pdf")
    plt.close()
    pass


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class TrainingSetGenerator:

    def __init__(self, N, Ws, n_max, repetitions):
        self.N = int(N)
        self.n_max = n_max
        self.repetitions = repetitions
        self.Ws = Ws
        self.training_set = self.generate_training_set_m_lanczos_list()  # self.generate_training_set_list()

    def generate_training_set_m_lanczos_list(self):
        """
        Returns training set with shape samples x [density matrix, W, lattice sites, block size, ground state energy]
        :return: training set
        """
        training_set = {consecutive_spins: [] for consecutive_spins in range(1, self.n_max + 1)}
        for rep in range(self.repetitions):
            for W in self.Ws:
                h = np.random.uniform(-W, W, size=self.N)
                H = gen_hamiltonian_lists(self.N, h, J=1)
                try:
                    Es, vs = eigsh(H, k=6, sigma=0, which='SM', tol=0.01)
                    # sigma=0 for shift invert mode Eigval near to zero
                    # following the advice of https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
                except ArpackNoConvergence as err:
                    Es = err.eigenvalues
                    vs = err.eigenvectors
                for i in range(len(Es)):
                    rho = np.outer(vs[:,i],vs[:,i])
                    for n in range(1, self.n_max + 1):
                        reduced_rho = self.get_partial_trace_first(rho, n) # fixme
                        # if Es[i]:
                        training_set[n].append([reduced_rho, W, self.N, n, Es[i], rep])
                        # else:
                        #     print("Diagonalization did not converge.")
        print(np.shape(training_set[n]), np.shape(training_set[n][0]))
        return training_set

    # def get_partial_trace_mid(self, rho, n):
    #     """
    #     calculates partial trace of middle n sites
    #     :param rho: full density matrix
    #     :param n: block size
    #     :return: reduced density matrix
    #     """
    #     kept_sites = self.get_keep_indices(n)
    #     qutip_dm = qutip.Qobj(rho, dims=[[2] * self.N] * 2)
    #     reduced_dm_via_qutip = qutip_dm.ptrace(kept_sites).full()
    #     return reduced_dm_via_qutip

    def get_partial_trace_first(self, rho, n):
        """
        calculates partial trace of first n sites
        :param rho: full density matrix
        :param n: block size
        :return: reduced density matrix
        """
        rho_ = rho.reshape((2 ** n, 2 ** (self.N - n), 2 ** n, 2 ** (self.N - n)))
        return np.einsum('jiki->jk', rho_)

    def diff(self, first, second):
        second = set(second)
        return [item for item in first if item not in second]

    def get_keep_indices(self, n):
        """
        Determines the middle indices for lattice sites numbered from 0 to N-1. Picks left indices more favourably.
        :return: List of complement of n consecutive indices
        """
        left_center = n // 2
        right_center = n - left_center
        middle = self.N // 2
        sites = np.arange(self.N)
        return sites[middle - left_center:middle + right_center].tolist()


if __name__ == "__main__":
    Ns = [9, 10, 11, 12]
    n_max = 6
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    repetitions = 10
    generate_training_set(Ns, Ws, n_max, repetitions)

    # N=09, n=7, rep=10 7s=> rep=500: 6 min
    # N=10, n=7, rep=10 31s => rep=500: 25 min
    # N=11, n=7, rep=10 182s=> rep=500: 2,5 h
    # N=12, n=7, rep=10 00s=> rep=500

    # N=8, n=7, rep=100, first, 26s
    # N=8, n=7, rep=100, mid, 32s
