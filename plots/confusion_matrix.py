import matplotlib.pyplot as plt
import numpy as np

import itertools
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import metrics



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
#cnf_matrix = confusion_matrix(y_test, y_pred)




labels_true = pickle.load(open("modelLuisaLabel/true_labels.p","rb")) 
predicted_labels = pickle.load(open("modelLuisaLabel/predicted_labels.p","rb"))


accuracy = metrics.adjusted_rand_score(labels_true, predicted_labels)  
print accuracy


cnf_matrix = confusion_matrix(labels_true, predicted_labels)
plot_confusion_matrix(cnf_matrix, classes="0123456789", title='Confusion matrix')


plt.savefig("confusion_matrix")
plt.show()





