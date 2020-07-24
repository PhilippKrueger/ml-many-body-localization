from ed import *
import time
from dataset_preparation import save_pickle
import qutip
from tqdm import trange, tqdm
from scipy.sparse.linalg import eigsh


def generate_training_set(Ns, Ws, n_max, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, Ws, n_max, repetitions)
        print("Training Set N=" + str(N) + " completed after %s seconds." % (time.time() - start_time))
        for n in range(1, n_max+1):
            save_pickle("lanczos/training_sets/N" + str(N) + "n" + str(n) + "_Trainset",
                    training_set_generator.training_set[n])
    print("--- Training set generation lasted %s seconds ---" % (time.time() - start_time))
    pass


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
        for rep in trange(self.repetitions):
            for W in self.Ws:
                Es, vs = self.get_ground_states(W)
                for i in range(len(Es)):
                    rho = np.outer(vs[:,i],vs[:,i])
                    for n in range(1, self.n_max + 1):
                        reduced_rho = self.get_partial_trace_mid(rho, n)
                        training_set[n].append([reduced_rho, W, self.N, n, Es[i], rep])
        return training_set

    def get_ground_states(self, W):
        hs = np.random.uniform(-W, W, size=self.N)
        # print(hs)
        H = gen_hamiltonian_lists(self.N, hs, J=-1)  # J defined as in original task
        try:
            Es, vs = eigsh(H, k=6, sigma=0, which='LM', tol=0.01)  # SM 1.4s, sigma=0, LM 5.2/s
            # sigma=0, 'LM' for shift invert mode Eigval near to zero
            # following the advice of https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        except:
            Es, vs = self.get_ground_states(W)
        return Es, vs

    def get_partial_trace_mid(self, rho, n):
        """
        calculates partial trace of middle n sites
        :param rho: full density matrix
        :param n: block size
        :return: reduced density matrix
        """
        kept_sites = self.get_keep_indices(n)
        qutip_dm = qutip.Qobj(rho, dims=[[2] * self.N] * 2)
        reduced_dm_via_qutip = qutip_dm.ptrace(kept_sites).full()
        return reduced_dm_via_qutip

    def get_partial_trace_first(self, rho, n):
        """
        calculates partial trace of first n sites
        :param rho: full density matrix
        :param n: block size
        :return: reduced density matrix
        """
        rho_ = rho.reshape((2 ** n, 2 ** (self.N - n), 2 ** n, 2 ** (self.N - n)))
        return np.einsum('jiki->jk', rho_)

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
    Ns = [8] # up to date: 9, 10, 11
    n_max = 6
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    repetitions = 100
    generate_training_set(Ns, Ws, n_max, repetitions)