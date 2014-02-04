'''
T.J. Bay - Training a neural network MAGIC telescope dataset using PyBrain

Discussion:  I decided to use the simulated MAGIC telescope dataset:
(http://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope).  
The dataset consists of 10 continuous features for 19020 observations and a label that is 
either 'g' for a gamma-ray event or an 'h' for a hadron event.  The telescope is
looking for gamma-ray events, so I labelled 'g' events as 1 and 'h' events as 0.

'''

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pybrain.datasets.supervised import SupervisedDataSet

from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.tools.customxml.networkwriter import NetworkWriter

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
    X = StandardScaler().fit_transform(X)
    X,Y = shuffle(X, Y, n_samples=None, random_state=None)

    N = len(Y)

    alldata = ClassificationDataSet(inp = 10, target = 1, nb_classes=2)

    for i in range(N):
        alldata.addSample(X[i],Y[i])

    tstdata, trndata = alldata.splitWithProportion(0.25)

    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]

    fnn = buildNetwork(trndata.indim, 15, trndata.outdim, bias=True) 
    trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0, learningrate = 0.005, verbose=True)

    trainer.trainUntilConvergence(maxEpochs=100)

    trnresult = 100 - percentError(trainer.testOnClassData(), trndata['class'])
    tstresult = 100 - percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

    print "Epoch:%4d" % trainer.totalepochs, \
      "  train accuracy: %5.2f%%" % trnresult, \
      "  test accuracy: %5.2f%%" % tstresult

    plt.figure(figsize=(8,8))
    plt.plot(trainer.trainingErrors)
    plt.xlabel('Training Steps')
    plt.ylabel('Training Error')
    plt.show()

    #NetworkWriter.writeToFile(fnn, 'magic_nn.xml')

if __name__ == "__main__":
    main()








