import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, \
    roc_auc_score, plot_roc_curve

m = 60000
m_test = 10000
digits = 10


def compute_loss(Y, Y_hat):
    m = Y.shape[1]
    L = -(1. / m) * (np.sum(np.multiply(np.log(Y_hat), Y)) + np.sum(np.multiply(np.log(1 - Y_hat), (1 - Y))))

    return L


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def main_function():
    a = []
    b = []
    X_train_images, X_train_labels = loadlocal_mnist(
        images_path='train-images-idx3-ubyte.idx3-ubyte',
        labels_path='train-labels-idx1-ubyte.idx1-ubyte')

    Y_test_images, Y_test_labels = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte.idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte.idx1-ubyte')

    X_train_images = X_train_images / 255
    Y_test_images = Y_test_images / 255

    X_train_labels = X_train_labels.reshape(1, 60000)
    X_new = np.eye(digits)[X_train_labels.astype('int32')]
    X_new = X_new.T.reshape(digits, 60000).T

    Y_test_labels = Y_test_labels.reshape(1, 10000)
    Y_new = np.eye(digits)[Y_test_labels.astype('int32')]
    Y_new = Y_new.T.reshape(digits, 10000).T

    n_x = 784
    n_h = 64
    learning_rate = 0.1

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(digits, n_h)
    b2 = np.zeros((digits, 1))
    correct = 0

    for i in range(1):

        for j in range(X_train_images.shape[0]):
            m = 1

            X = np.array(X_train_images[j]).reshape(1, 784)
            Y = np.array(X_new[j]).reshape(10, 1)

            Z1 = np.matmul(W1, X.T) + b1

            A1 = sigmoid(Z1)

            Z2 = np.matmul(W2, A1) + b2

            A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

            cost = compute_loss(Y, A2)

            print("Cost is " + str(cost))

            dZ2 = A2 - Y

            dW2 = (1. / m) * np.matmul(dZ2, A1.T)
            db2 = (1. / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.matmul(W2.T, dZ2)
            dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))

            dW1 = (1. / m) * np.matmul(dZ1, X)
            db1 = (1. / m) * np.sum(dZ1, axis=1, keepdims=True)

            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1

            print("Epoch", i, "cost: ", cost)

        print("Final cost:", cost)
        a.append(i)
        b.append(cost)
    plt.plot(a, b)

    # Labelling the x axis as the iterations axis.
    plt.xlabel("Iterations")

    # Labelling the y axis as the cost axis.
    plt.ylabel("Cost")

    # Showing the plot.
    plt.show()

    c = []
    d = []
    for k in range(10000):

        X1 = Y_test_images[k].reshape(1, 784)
        Y1 = Y_new[k].reshape(10, 1)

        Z1 = np.matmul(W1, X1.T) + b1

        A1 = sigmoid(Z1)

        Z2 = np.matmul(W2, A1) + b2

        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

        cost = compute_loss(Y1, A2)

        print("Cost is " + str(cost))

        val1 = np.argmax(A2, axis=0)
        c.append(val1)
        val2 = np.argmax(Y1, axis=0)
        d.append(val2)
        if val1 == val2:
            correct = correct + 1

    print(correct * 100 / 10000)

    print(confusion_matrix(c, d))
    print(classification_report(c, d))
    # print(roc_curve(c, d))

    auc = roc_auc_score(c, d, multi_class="ovo", average="weighted")
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(c, d)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


main_function()
