# What are you trying to solve?
# How did I go about solving it?
# What is the result
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve, GridSearchCV
from mlxtend.plotting import plot_learning_curves
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from statistics import plot_learning_curve
from utility import *


def main():
    images_train, labels_train = load_mnist("./datasets", "train")
    images_test, labels_test = load_mnist("./datasets", "t10k")
    print("hey there, time to train data!")


    classifier = MLPClassifier(hidden_layer_sizes=(750,), activation='logistic', solver='adam', max_iter=500)

    print("Fitting data...")
    classifier.fit(images_train, labels_train)
    #
    # print("Generating confusion matrix...")
    # ConfusionMatrixDisplay.from_estimator(
    #     classifier, images_test, labels_test)
    #
    # plt.show()

    #Find the best hidden Layer Count

    ### RUN GRID SEARCH FOR HIDDEN LAYERS ###
    # classifier_testLayers = MLPClassifier(activation='logistic', solver='adam', max_iter=500)
    # print(classifier_testLayers.get_params().keys())
    # hiddenLayerRange = np.linspace(50, 1000, num=5, dtype=int)
    #
    # parameter_space = {
    #     'hidden_layer_sizes': [(20,), (100,), (500,), (750,), (1000,)],
    # }
    #
    # print("Doing grid search...")
    # gsClf = GridSearchCV(classifier_testLayers, parameter_space, n_jobs=-1, cv=3)
    # gsClf.fit(images_train, labels_train)
    # # Best parameter set
    # print('Best parameters found:\n', gsClf.best_params_)
    #
    # # All results
    # means = gsClf.cv_results_['mean_test_score']
    # stds = gsClf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, gsClf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    #
    # print("Predicting for the best results...")
    # y_true, y_pred = labels_test, gsClf.predict(images_test)
    #
    # from sklearn.metrics import classification_report
    # print('Results on the test set:')
    # print(classification_report(y_true, y_pred))

    ### END GRID SEARCH ###

    print("Predicting on test data...")
    labels_prediction = classifier.predict(images_test)

    print("Generating score...")
    score = accuracy_score(labels_test, labels_prediction)
    print('Accuracy Score: ' + str(score * 100) + '%')
    #


if __name__ == "__main__":
    main()



