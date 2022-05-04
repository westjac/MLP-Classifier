# What are you trying to solve?
# How did I go about solving it?
# What is the result
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from statistics import *
from utility import *


def main():
    images_train, labels_train = load_mnist("./datasets", "train")
    images_test, labels_test = load_mnist("./datasets", "t10k")
    print("hey there, time to train data!")

    classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam') # 784

    #print("Fitting data...")
    #classifier.fit(images_train, labels_train)

    #accuracy_vs_hiddenLayers(images_train, labels_train, images_test, labels_test)

    # print("Predicting on test data...")
    # labels_prediction = classifier.predict(images_test)
    #
    # print("Generating score...")
    # score = accuracy_score(labels_test, labels_prediction)
    # print('Accuracy Score: ' + str(score * 100) + '%')
    #
    # print("Generating confusion matrix...")
    # confusion = confusion_matrix(labels_test, labels_prediction)
    # print(confusion)

    train_sizes, train_scores, test_scores = learning_curve(classifier, images_train, labels_train)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplots(1, figsize=(10,10))
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()



