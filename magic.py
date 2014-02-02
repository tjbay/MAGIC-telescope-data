'''
T.J. Bay - Comparison of classifiers for the MAGIC telescope dataset

Discussion:  I decided to use the simulated MAGIC telescope dataset:
(http://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope).  
The dataset consists of 10 continuous features for 19020 observations and a label that is 
either 'g' for a gamma-ray event or an 'h' for a hadron event.  The telescope is
looking for gamma-ray events, so I labelled 'g' events as 1 and 'h' events as 0.

For low values of C ( C < 1.3, strong regularization), the accuracy goes down.  This indicates underfitting.
For high vlaues of C (weak regularization), the accuracy basically stays the same, with perhaps a slight
decrease for C > 25.  I interpret this to mean that there is sufficient data to prevent overfitting.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


#test logistic, knn, gaussian nb, random forest

def load_magic_data():
    '''
    1.  fLength:  continuous  # major axis of ellipse [mm]
    2.  fWidth:   continuous  # minor axis of ellipse [mm] 
    3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
    4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
    5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
    6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
    7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
    8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
    9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
    10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
    11.  class:    g,h         # gamma (signal), hadron (background)
    '''

    Ynames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
             'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'identity']

    featureNames = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
             'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

    filepath = 'data/magic04.data'

    data = pd.read_csv(filepath, names=Ynames, header=None)
 
    data['identity'][data['identity'] == 'g'] = 1
    data['identity'][data['identity'] == 'h'] = 0

    X = data[featureNames].values
    Y = data['identity'].values.astype('int64')

    return (X,Y,Ynames)


def main():

    (X, Y, Ynames) = load_magic_data()
    
    #X,Y = shuffle(X, Y, n_samples = 100, random_state=0)

    n_samples, n_features = X.shape
    C = 2.0

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=None)

    classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic': LogisticRegression(C=C, penalty='l2'),
               'KNN': KNeighborsClassifier(n_neighbors=11),
               'NB': GaussianNB(),
               'RF3': RandomForestClassifier(n_estimators=3),
               'RF25': RandomForestClassifier(n_estimators=25),
               'RF100': RandomForestClassifier(n_estimators=100)}

    n_classifiers = len(classifiers)

    plt.figure(figsize=(8,8))

    for index, (name, clf) in enumerate(classifiers.iteritems()):
        clf.fit(Xtrain, Ytrain)
        probs = clf.predict_proba(Xtest)

        fpr, tpr, thresholds = roc_curve(Ytest, probs[:, 1])
        roc_auc = auc(fpr, tpr)

    
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (name, roc_auc))


    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()