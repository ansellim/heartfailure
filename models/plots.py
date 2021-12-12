import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    # TODO: Make plots for loss curves and accuracy curves.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    # print(train_losses)
    plt.figure()
    plt.title('loss curve')
    plt.plot(train_losses, label='train_losses')
    plt.plot(valid_losses, label='valid_losses')
    plt.legend()
    plt.savefig('loss_curve.jpg')
    plt.close()

    plt.figure()
    plt.title('accuracy curve')
    plt.plot(train_accuracies, label='train_accuracies')
    plt.plot(valid_accuracies, label='valid_accuracies')
    plt.legend()
    plt.savefig('accuracy_curve.jpg')
    plt.close()


# pass


def plot_confusion_matrix(results, class_names):
    # TODO: Make a confusion matrix plot.
    # TODO: You do not have to return the plots.
    # TODO: You can save plots as files by codes here or an interactive way according to your preference.
    y_true = [i[0] for i in results]
    y_pred = [i[1] for i in results]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('confusion_matrix.jpg')
    plt.close()
# print(results)
# print(class_names)
# plt.figure()
# plot_confusion_matrix(clf, X_test, y_test)
# plt.savefig('accuracy_curve.jpg')
# plt.close()
# pass
