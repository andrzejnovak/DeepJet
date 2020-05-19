import sys
import os
import keras
import tensorflow as tf

from keras.losses import kullback_leibler_divergence, categorical_crossentropy
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
from Metrics import * #needed!
import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import h5py
from Losses import NBINS

#sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess = tf.InteractiveSession()

def loadModel(inputDir,trainData,model,LoadModel,sampleDatasets=None,removedVars=None,adv=False):
    inputModel = '%s/KERAS_check_best_model.h5'%inputDir

    from DeepJetCore.DataCollection import DataCollection
    traind=DataCollection()
    traind.readFromFile(trainData)
    traind.dataclass.regressiontargetclasses = range(0,NBINS)
    print(traind.getNRegressionTargets())

    if(LoadModel):
        evalModel = load_model(inputModel, custom_objects = global_loss_list)
        shapes=traind.getInputShapes()

    else:
        shapes=traind.getInputShapes()
        train_inputs = []
        for s in shapes:
            train_inputs.append(keras.layers.Input(shape=s))
        modelargs = {}
        if adv:
            modelargs.update({'nRegTargets':NBINS,
                              'discTrainable': True,
                              'advTrainable':True})
        evalModel = model(train_inputs,traind.getNClassificationTargets(),traind.getNRegressionTargets(),sampleDatasets,removedVars,**modelargs)
        evalModel.load_weights(inputModel)

    return evalModel

def evaluate(testd, trainData, model, outputDir, storeInputs=False, adv=False, tlimit=None):
    """
    tlimit ~ time limit to stop in minutes

    """
    NENT = 20  # Chunk it
    filelist = []

    feature_names = testd.dataclass.branches[1]
    spectator_names = testd.dataclass.branches[0]

    from DeepJetCore.DataCollection import DataCollection
    traind = DataCollection()
    traind.readFromFile(trainData)
    truthnames = traind.getUsedTruth()
    cols = spectator_names + ['predict' + tn for tn in truthnames
                              ] + ['truth' + tn for tn in truthnames]
    df = pd.DataFrame([], columns=cols)

    import time
    tic = time.time()

    for i, s in enumerate(testd.samples):
        spath = testd.getSamplePath(s)
        filelist.append(spath)
        h5File = h5py.File(spath)

        for ni in range(NENT):
            print('Processing chunk - {}/{} '.format(((i) * NENT + ni),
                                                     len(testd.samples) * NENT))
            features_val = [
                h5File['x%i' % j][()][ni::NENT]
                for j in range(0, h5File['x_listlength'][()][0])
            ]
            if adv:
                predict_test_i = model.predict(features_val, batch_size=8192)[:, NBINS:]
            else:
                predict_test_i = model.predict(features_val, batch_size=8192)
            labels_val_i = h5File['y0'][()][ni::NENT, :]
            spectators_val_i = h5File['z0'][()][ni::NENT, 0, :]

            #tdf = pd.DataFrame(spectators_val_i)
            print(spectators_val_i.shape)
            print(labels_val_i.shape)
            print(predict_test_i.shape)
            tnp = np.concatenate([spectators_val_i, predict_test_i, labels_val_i],
                                 axis=1)
            tdf = pd.DataFrame(tnp, columns=cols)
            df = df.append(tdf)

            # Some timing
            frc = ((i) * NENT + ni) / float((len(testd.samples) * NENT)) + 0.0001
            lapsed = time.time() - tic
            tot_time = lapsed / frc
            print("ETA: {0:.2f} mins".format((tot_time - lapsed) / 60))

            if df.shape[0] > 3000000:
                break

            if tlimit is not None:    
                if lapsed > 60*tlimit:
                    break

    print("Testing prediction:")
    print("Total: ", len(df))
    for lab in truthnames:
        print(lab, ":", sum(df['truth' + lab].values))

    df.to_pickle(outputDir + '/output.pkl')

    return df
