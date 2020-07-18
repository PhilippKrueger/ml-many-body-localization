from generate_training_set import TrainingSetGenerator, save_pickle
from model_save_train import *
from scipy.optimize import curve_fit


def preprocess_test_data(path):
    data = load_pickle(path)
    X = [item[0] for item in data]
    X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1))
    X = np.asarray(np.concatenate((np.real(X), np.imag(X)), axis=3))
    W = np.reshape(np.asarray([item[1] for item in data]), (np.shape(data)[0], 1))
    return X, W


def logistic(x, a):
    return 1 / (1 + np.exp(-100 * (x - a)))


def heaviside(x, a):
    return 0.5*np.sign(x-a)+0.5


def load_model(path):
    return models.load_model(path)


def get_wc(N, n, W_max):
    model = load_model('lanczos/models/N' + str(N) + 'n' + str(n) + '_Model')
    X, W = preprocess_test_data('lanczos/test_sets/N' + str(N) + 'n' + str(n) +'_Testset')
    state_prediction = model.predict(X)


    # print(W_max, np.shape(W_max))
    popt, pcov = curve_fit(logistic, W_max, np.reshape(state_prediction, (len(state_prediction))))  # state_prediction.astype(np.float))

    # fixme plot all n together!!
    fig, ax1 = plt.subplots()
    ax1 = plt.scatter(W_max, state_prediction)
    ax1 = plt.plot(W_max, logistic(W_max, *popt), 'k')

    plt.title('Phase prediction $N = $' + str(N) + ", $W_c = $" + "{0:.3g}".format(popt[0]))
    plt.ylabel('Probability of localized phase')
    plt.xlabel('$W_{max}$')
    plt.legend(['Logistic fit', 'Predicted phase'], loc='upper left')
    plt.savefig('results/N' + str(N) + '_predict_wc.pdf')
    return popt[0], N, np.shape(X[0])[0]


def plot_wc_dependencies(Ns, W_max):
    Wc_dependencies = []
    for N in Ns:
        for n in range(1, N+1):
            Wc_dependencies.append(get_wc(N, n, W_max))
    Wc_dependencies = np.array([np.array(xi) for xi in Wc_dependencies])
    print(Wc_dependencies)
    fig, ax1 = plt.subplots()
    plt.title("$W_c$ dependency on system size L")
    plt.ylabel('Critical maximum disorder strength $W_c$')
    plt.xlabel('System size $L$')
    ax1 = plt.scatter(Wc_dependencies[:, 1], Wc_dependencies[:, 0])
    plt.savefig('results/Wc_L_dependency.pdf')
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("$W_c$ dependency on block size n")
    plt.ylabel('Critical maximum disorder strength $W_c$')
    plt.xlabel('Block size $n$')
    ax1 = plt.scatter(Wc_dependencies[:, 2], Wc_dependencies[:, 0])
    plt.savefig('results/Wc_n_dependency.pdf')
    plt.close()


if __name__ == "__main__":
    Ns = [9, 10]
    W_max = np.arange(0., 4.0, 0.05)
    plot_wc_dependencies(Ns, W_max)
