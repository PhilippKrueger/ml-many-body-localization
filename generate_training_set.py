from ed import *
import time
import pickle
from scipy.sparse.linalg import ArpackNoConvergence
import qutip


def generate_training_set(Ns, Ws, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, Ws, repetitions)
        print("Training Set N="+str(N)+" completed after %s seconds." % (time.time() - start_time))
        for n in range(1,N+1):
            save_groundstate_figures(N, training_set_generator.training_set[n], n)
            save_pickle("lanczos/training_sets/N" + str(N) + "n" + str(n) + "_Trainset", training_set_generator.training_set[n])
    print("--- Training set generation lasted %s seconds ---" % (time.time() - start_time))
    pass


def save_groundstate_figures(N, training_set, n): # reduced_rho, W, self.N, n, E
    ergodic = [item for item in training_set if item[1] == 0.5 and item[-1] == 0][0] # len: repetitions
    localized = [item for item in training_set if item[1] == 8 and item[-1] == 0][0] # len: repetitions

    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(ergodic[0]))
    fig.colorbar(pos, ax=ax1)
    plt.title("Reduced density matrix for $n=$" + str(n) + " consecutive sites \n at $E=$"
              + str(round(ergodic[4], 2)) + " for $W=$" + str(ergodic[1]) + ", $N = $" + str(N))
    plt.savefig(
        "results/groundstates/N" + str(N) + "n" + str(n) + "_trainingset_groundstate_Wmax" + str(ergodic[1]) + ".pdf")
    plt.close()

    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(localized[0]))
    fig.colorbar(pos, ax=ax1)
    plt.title("Reduced density matrix for $n=$" + str(localized[3]) + " consecutive sites \n at $E=$"
              + str(round(localized[4], 2)) + " for $W=$" + str(localized[1]) + ", $N = $" + str(N))
    plt.savefig(
        "results/groundstates/N" + str(N) + "n" + str(localized[3]) + "_trainingset_groundstate_Wmax" + str(localized[1]) + ".pdf")
    plt.close()
    pass


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class TrainingSetGenerator:

    def __init__(self, N, Ws, repetitions):
        self.N = int(N)  # Lattice sites
        self.repetitions = repetitions
        self.Ws = Ws
        self.training_set = self.generate_training_set_m_lanczos_list()  # self.generate_training_set_list()

    def generate_training_set_m_lanczos_list(self):
        """
        Returns training set with shape samples x [density matrix, W, lattice sites, block size, ground state energy]
        :return: training set
        """
        training_set = {consecutive_spins: [] for consecutive_spins in range(1,self.N+1)}
        for W in self.Ws:
            for rep in range(self.repetitions):
                H = gen_hamiltonian_random_h(self.N, W=W, J=1.)
                E, v = qutip.Qobj(H).groundstate()
                rho = np.outer(v, v) #vs[:, 0]
                for n in range(1, self.N):
                    reduced_rho = self.get_partial_trace(rho, n) # must trace out something
                    training_set[n].append([reduced_rho, W, self.N, n, E, rep])
                training_set[self.N].append([rho, W, self.N, self.N, E, rep])
        return training_set

    def get_partial_trace(self, rho, n):
        """
        calculates partial trace by reshaping the density matrix and adding along the axis
        :param rho: full density matrix
        :param n: block size
        :return: reduced density matrix
        """
        kept_sites = self.get_keep_indices(n)
        qutip_dm = qutip.Qobj(rho, dims=[[2]*self.N]*2)
        reduced_dm_via_qutip = qutip_dm.ptrace(kept_sites).full()
        return reduced_dm_via_qutip

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
    Ns = [10]
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    repetitions = 500  # 55 min
    generate_training_set(Ns, Ws, repetitions)

