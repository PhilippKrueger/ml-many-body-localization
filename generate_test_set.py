from generate_training_set import TrainingSetGenerator, save_pickle
from model_save_train import *
import time


def generate_test_set(Ns, W_max, repetitions):
    start_time = time.time()
    for N in Ns:
        training_set_generator = TrainingSetGenerator(N, W_max, repetitions)
        save_pickle("pickled/N"+str(N)+"_Testset", training_set_generator.training_set)
    print("--- Testing set generation lasted %s seconds ---" % (time.time() - start_time))
    pass


if __name__ == "__main__":
    Ns = [9, 10, 11, 12]
    W_max = np.arange(0., 4.0, 0.05)
    repetitions = 1
    generate_test_set(Ns, W_max, repetitions)

    # N = 10 <70s
    # N = 11 70s
    # N = 12 163s

