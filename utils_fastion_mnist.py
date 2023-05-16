import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

labelNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion_mnist_data():
    ((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

    # Normalize the data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0

    print(x_train.shape)
    print(x_test.shape)

    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist_data_reshape():
    ((x_train, y_train), (x_test, y_test)) = load_fashion_mnist_data()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    return (x_train, y_train), (x_test, y_test)


# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labelNames))
    plt.xticks(tick_marks, labelNames, rotation=90)
    plt.yticks(tick_marks, labelNames)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def generate_report(module_name, y_pred_clf, y_test):
    print("---------------- " + module_name + " Report ---------------")
    clf_f1 = metrics.f1_score(y_test, y_pred_clf, average="weighted")
    clf_accuracy = metrics.accuracy_score(y_test, y_pred_clf)
    clf_cm = metrics.confusion_matrix(y_test, y_pred_clf)
    print("F1 score: {}".format(clf_f1))
    print("Accuracy score: {}".format(clf_accuracy))
    print("Confusion matrix: \n", clf_cm)

    plt.figure()
    plot_confusion_matrix(clf_cm)
    plt.show()

    print(metrics.classification_report(y_test, y_pred_clf))

