# What are you trying to solve?
# How did I go about solving it?
# What is the result
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from utility import *


def main():
    images_train, labels_train = load_mnist("./datasets", "train")
    images_test, labels_test = load_mnist("./datasets", "t10k")
    print("hey there, time to train data.")

    classifier = MLPClassifier(hidden_layer_sizes=(784,), activation='logistic', solver='adam')

    classifier.fit(images_train, labels_train)

    labels_prediction = classifier.predict(images_test)

    score = accuracy_score(labels_test, labels_prediction)
    print(score)


if __name__ == "__main__":
    main()



