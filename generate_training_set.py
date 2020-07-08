from ed_conserve import *
import random
import time
import pickle


class TrainingSetGenerator:

    def __init__(self, Ls, Ns, Wtrain):
        self.Ls = Ls
        self.Ns = Ns
        self.hs = [self.get_disorder_strength(Wtrain)]
        self.training_set = self.generate_training_set()

    def get_disorder_strength(self, Wtrain):
        # h = np.random.uniform(-Wtrain, Wtrain)
        return random.choice(Wtrain)

    def generate_training_set(self):
        training_set = []
        for N in self.Ns:
            for h in self.hs:
                E0, v0, k0 = self.calc_lowest_eigenstate(N, h)
                rho = self.calc_density_matrix(v0)
                training_set.append([rho, h, N, E0, v0, k0])#fixme L dependency?
        return training_set

    def calc_lowest_eigenstate(self, N, h):
        H = calc_H(N, J=1., g=h)
        plt.figure()
        for qn in H:
            k = qn * 2 / 14
            E, v = scipy.sparse.linalg.eigsh(H[qn], k=1, which='SA')#fixme k=0
        return E[0], v[:, 0], k

    def calc_density_matrix(self, v):
        return np.outer(v, v)


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        # done





if __name__ == "__main__":
    Ls = [10, 11, 12]#fixme unused
    start_time = time.time()
    Ns = [10]*10
    W_train = [0.5, 8.0]
    training_set_generator = TrainingSetGenerator(Ls, Ns, W_train)
    print("--- %s seconds ---" % (time.time() - start_time))

    save_pickle("pickled/N"+str(10), training_set_generator.training_set)

    print(training_set_generator.training_set[0][0])
    plt.imshow(np.real(training_set_generator.training_set[0][0]))
    plt.savefig("results/test.jpg")

    # print(len(training_set_generator.training_set))
    # print(np.shape(training_set_generator.training_set[0][0]))

