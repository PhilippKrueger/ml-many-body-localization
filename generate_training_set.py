from ed_conserve import *
import random
import time
import pickle


def get_disorder_strength(Wtrain):
    # h = np.random.uniform(-Wtrain, Wtrain)
    return random.choice(Wtrain)


def get_random_disorder_strength(Wtrain):
    h = np.random.uniform(-Wtrain, Wtrain)
    return h


class TrainingSetGenerator:

    def __init__(self, Ns, Ws):
        self.Ns = Ns # Lattice sites
        self.Ws = Ws
        self.training_set = self.generate_training_set()

    def generate_training_set(self):
        training_set = []
        for N in self.Ns:
            for W in self.Ws:
                E0, v0, k0 = self.calc_lowest_eigenstate(N, W)
                rho = self.calc_density_matrix(v0)
                training_set.append([rho, W, N, E0, v0, k0])
        return training_set

    def calc_lowest_eigenstate(self, N, h):
        H = calc_H_random_field(N, J=1., Wmax=h)
        for qn in H:
            k = qn * 2 / 14
            E, v = scipy.sparse.linalg.eigsh(H[qn], k=1, which='SA')#fixme k=0
        return E[0], v[:, 0], k

    def calc_density_matrix(self, v):
        return np.outer(v, v)


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    start_time = time.time()
    Ns = [10]*100
    Ws = [0.5, 8.0]
    training_set_generator = TrainingSetGenerator(Ns, Ws)
    print("--- %s seconds ---" % (time.time() - start_time))

    save_pickle("pickled/N10_Trainset", training_set_generator.training_set)

    print(training_set_generator.training_set[0][0])
    plt.imshow(np.real(training_set_generator.training_set[0][0]))
    plt.savefig("results/test.jpg")