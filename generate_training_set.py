from ed import *
import time
import pickle
from scipy.sparse.linalg import ArpackNoConvergence
import qutip


def generate_training_set(Ns, Ws, M, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, Ws, M, repetitions)
        print("Training Set N="+str(N)+" completed after %s seconds." % (time.time() - start_time))
        save_pickle("lanczos/N" + str(N) + "_Trainset", training_set_generator.training_set)
        save_groundstate_figures(N, training_set_generator.training_set)
        # save_eigenvalue_dispersion(N, M, training_set_generator.training_set)
    print("--- Training set generation lasted %s seconds ---" % (time.time() - start_time))
    pass


def save_eigenvalue_dispersion(N, M, training_set):
    ergodic = np.asarray([item for item in training_set if item[1] == 0.5])
    localized = np.asarray([item for item in training_set if item[1] == 8])
    fig, ax1 = plt.subplots()
    ax1 = plt.plot(localized[:, 5], np.real(localized[:, 3]), 'bo')
    ax1 = plt.title(
        "Lowest " + str(M) + " states per k for $W_{max}=$" + str(localized[0, 1]) + " , $N = $" + str(localized[0, 2]))
    plt.xlabel('$k/\\pi$')
    plt.ylabel('$E(k)$')
    plt.savefig(
        "results/N" + str(N) + "_trainingset_lowest_m_states_Wmax_localized.pdf")

    fig, ax1 = plt.subplots()
    ax1 = plt.plot(ergodic[:, 5], np.real(ergodic[:, 3]), 'bo')
    ax1 = plt.title(
        "Lowest " + str(M) + " states per k for $W_{max}=$" + str(ergodic[0, 1]) + " , $N = $" + str(
            ergodic[0, 2]))
    plt.xlabel('$k/\\pi$')
    plt.ylabel('$E(k)$')
    plt.savefig(
        "results/N" + str(N) + "_trainingset_lowest_m_states_Wmax_ergodic.pdf")
    pass


def save_groundstate_figures(N, training_set): # reduced_rho, W, self.N, n, E
    ergodics = np.asarray([item for item in training_set if item[1] == 0.5 and item[-1] == 0])
    localizeds = np.asarray([item for item in training_set if item[1] == 8 and item[-1] == 0])

    for ergodic in ergodics:
        fig, ax1 = plt.subplots()
        pos = ax1.imshow(np.real(ergodic[0]))
        fig.colorbar(pos, ax=ax1)
        plt.title("Reduced density matrix for $n=$" + str(ergodic[3]) + " consecutive sites \n at $E=$"
                  + str(round(ergodic[4], 2)) + " for $W_{max}=$" + str(ergodic[1]) + ", $N = $" + str(N))
        plt.savefig(
            "results/groundstates/N" + str(N) + "n" + str(ergodic[3]) + "_trainingset_groundstate_Wmax" + str(ergodic[1]) + ".pdf")
        plt.close()

    for localized in localizeds:
        fig, ax1 = plt.subplots()
        pos = ax1.imshow(np.real(localized[0]))
        fig.colorbar(pos, ax=ax1)
        plt.title("Reduced density matrix for $n=$" + str(localized[3]) + " consecutive sites \n at $E=$"
                  + str(round(localized[4], 2)) + " for $W_{max}=$" + str(localized[1]) + ", $N = $" + str(N))
        plt.savefig(
            "results/groundstates/N" + str(N) + "n" + str(localized[3]) + "_trainingset_groundstate_Wmax" + str(localized[1]) + ".pdf")
        plt.close()
    pass


def get_random_disorder_strength(Wtrain):
    h = np.random.uniform(-Wtrain, Wtrain)
    return h


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def calc_density_matrix(v):
    return np.outer(v, v)


class TrainingSetGenerator:

    def __init__(self, N, Ws, M, repetitions):
        self.M = M
        self.N = int(N)  # Lattice sites
        self.repetitions = repetitions
        self.Ws = Ws
        self.training_set = self.generate_training_set_m_lanczos_list()  # self.generate_training_set_list()

    def generate_training_set_m_lanczos_list(self):
        """
        Returns training set with shape samples x [density matrix, W, lattice sites, block size, ground state energy]
        :return: training set
        """
        training_set = []
        for W in self.Ws:
            for rep in range(self.repetitions):
                H = gen_hamiltonian_random_h(self.N, W=W, J=1.)
                E, v = qutip.Qobj(H).groundstate()
                rho = calc_density_matrix(v) #vs[:, 0]
                for n in range(1, self.N):
                    reduced_rho = self.get_partial_trace(rho, n) # must trace out something
                    training_set.append([reduced_rho, W, self.N, n, E, rep])
                training_set.append([rho, W, self.N, self.N, E, rep])
        return training_set

    def get_partial_trace(self, rho, n):
        """
        calculates partial trace by reshaping the density matrix and adding along the axis
        :param rho: full density matrix
        :param n: block size
        :return: reduced density matrix
        """
        kept_sites = self.get_keep_indices(n)
        # print(self.N,kept_sites)
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
    Ns = [9, 10, 11, 12]
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    M = 3  # Number of lowest eigenvalues to append to the training set per k
    repetitions = 1  # 55 min
    generate_training_set(Ns, Ws, M, repetitions)

    # training_set_generator = TrainingSetGenerator(N=3, Ws=[1,8], M=1, repetitions=1)
    # v = [0,0,1,0,0,0,1,0]
    # print("one last time")
    # training_set_generator.get_partial_trace(np.outer(v,v), 1)

