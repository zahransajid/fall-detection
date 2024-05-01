from sklearn import metrics
import itertools
import json
import numpy as np

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

with open("results.json") as f:
    data = json.load(f)
y_test = data["y_test"]
y_pred = data["y_pred"]

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['ADLs', 'Falls', 'Near_Falls'])

