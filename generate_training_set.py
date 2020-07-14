from ed_conserve import *
import random
import time
import pickle


def generate_training_set(Ns, Ws, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, Ws, repetitions)
        save_pickle("pickled/N"+str(N)+"_Trainset", training_set_generator.training_set)
        save_groundstate_figures(N, training_set_generator.training_set[0], training_set_generator.training_set[1])#fixme bad for performance
    print("--- Training set generation lasted %s seconds ---" % (time.time() - start_time))
    pass

def save_groundstate_figures(N, erogdic, localized):
    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(erogdic[0]))
    fig.colorbar(pos, ax=ax1)
    plt.title("Groundstate for $W_{max}=$"+str(erogdic[1])+" , $N = $"+str(N))
    plt.savefig(
        "results/N" + str(N) + "_trainingset_groundstate_Wmax" + str(erogdic[1]) + ".pdf")

    fig, ax1 = plt.subplots()
    pos = ax1.imshow(np.real(localized[0]))
    fig.colorbar(pos, ax=ax1)
    plt.title("Groundstate for $W_{max}=$" + str(erogdic[1]) + " , $N = $" + str(N))
    plt.savefig(
        "results/N" + str(N) + "_trainingset_groundstate_Wmax" + str(localized[1]) + ".pdf")
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

    def __init__(self, N, Ws, repetitions):
        self.N = int(N)  # Lattice sites
        self.repetitions = repetitions
        self.Ws = Ws
        self.training_set = self.generate_training_set()

    def generate_training_set(self):
        training_set = []
        for _ in range(self.repetitions):
            for W in self.Ws:
                E0, v0, k0 = self.calc_lowest_eigenstate(self.N, W)
                rho = calc_density_matrix(v0)
                training_set.append([rho, W, self.N, E0, v0, k0])
        return training_set

    def calc_lowest_eigenstate(self, N, h):
        H = calc_H_random_field(N, J=1., Wmax=h)
        for qn in H:
            k = qn * 2 / 14
            E, v = scipy.sparse.linalg.eigsh(H[qn], k=1, which='SA')  # fixme k=0
        return E[0], v[:, 0], k


if __name__ == "__main__":
    # Ns = [10, 11, 12]
    # Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    # repetitions = 100
    # generate_training_set(Ns, Ws, repetitions)


    Ns = [12]
    Ws = [0.5, 8.0]  # 0.5 => ergodic/delocalized phase, 8.0 localized phase
    repetitions = 100
    generate_training_set(Ns, Ws, repetitions)

    # N=10 Training set generation lasted 80.94452285766602 seconds
    # N=11 Training set generation lasted 167.01501274108887 seconds
    # N=12 Training set generation lasted 372.91300201416016 seconds