from modules import *
from sklearn.model_selection import StratifiedKFold
from keras.layers import BatchNormalization
from sklearn.metrics import log_loss, accuracy_score
import cPickle as pickle
import numpy as np
from scipy.io import loadmat,savemat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from tqdm import tqdm
import csv
import pandas as pd
import matplotlib.pyplot as plt
from dataexp import *

def stratified_split3(data,label):
    data3 = []
    label3 = []
    skf = StratifiedKFold(n_splits=3)
    skf.get_n_splits(data,label)
    for a in skf.split(data,label):
        (x,y)=a
        break
    data3.append(data[y])
    label3.append(label[y])

    temp_data = data[x]
    temp_label = label[x]
    skf = StratifiedKFold(n_splits=2)
    skf.get_n_splits(temp_data,temp_label)
    for a in skf.split(temp_data,temp_label):
        (x,y)=a
        break
    data3.append(temp_data[x])
    label3.append(temp_label[x])
    data3.append(temp_data[y])
    label3.append(temp_label[y])
    return data3,label3

def prepare_data(x,y):
    x = x.astype('float32')
    #x = x-np.mean(x,axis=0)
    #x=x/np.std(x,axis=0)
    n = len(y)
    shuf = np.arange(n)
    np.random.shuffle(shuf)
    x = x[shuf]
    y = y[shuf]
    return x,y

def expand_data(x,y):
    x_exp,y_exp = dataexp(x,y)
    return prepare_data(x_exp,y_exp)

def ann(data,label):
    data3,label3 = stratified_split3(data,label)
    nb_classes = int(label.max()+1)
    lr = 1e-3
    input_shape = data.shape[1]
    nb_epoch = 50
    actv_fn = 'relu'
    batch_size = 8
    adam = Adam(lr=lr)
    total_iter = 3

    labels_list = []
    preds_list = []
    probas_list = []
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):

        # get training, testing data with data balancing
        x_train = np.concatenate((data3[n_iter%3],data3[(n_iter+1)%3]),axis=0)
        x_test = data3[(n_iter+2)%3]
        y_train = np.concatenate((label3[n_iter%3],label3[(n_iter+1)%3]),axis=0)
        y_test = label3[(n_iter+2)%3]

        # prepare the data for efficient modelling
        x_train,y_train = expand_data(x_train,y_train)
        x_test,y_test = prepare_data(x_test,y_test)

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        # model the network
        # network : 100-100-100-9
        model = Sequential()
        model.add(Dense(100,input_dim=input_shape,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation(actv_fn))
        model.add(Dropout(0.8))

        model.add(Dense(100,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation(actv_fn))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        # train the network
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
        out,wt = model.fit(x_train,Y_train,batch_size=batch_size, epochs=nb_epoch, verbose=2 ,validation_data=(x_test, Y_test))
        best_wt = wt[np.argmin(out.history['val_loss'])]
        model.set_weights(best_wt)

        probas = model.predict_proba(x_test)
        pred_indices = np.argmax(probas, axis=1)
        preds = np.unique(y_test)[pred_indices]

        if n_iter==0:
            labels_list = y_test.copy()
            preds_list = preds.copy()
            probas_list = probas.copy()
        else:
            labels_list = np.concatenate((labels_list,y_test.copy()),axis=0)
            preds_list = np.concatenate((preds_list,preds.copy()),axis=0)
            probas_list = np.concatenate((probas_list,probas.copy()),axis=0)
        #plot_confmatrix(y_test, preds,normalize=True)
        #plt.title('Test confusion matrix')

        #probas1 = model.predict_proba(x_train)
        #pred_indices1 = np.argmax(probas1, axis=1)
        #preds1 = np.unique(y_train)[pred_indices1]
        #plot_confusion_matrix(y_train, preds1,normalize=True)
        #plt.title('Train confusion matrix')
        #print log_loss(y_test,probas),log_loss(y_train,probas1)
    return labels_list,preds_list,probas_list

def ann_clf(x_train,y_train,x_test,y_test):
    nb_classes = int(y_train.max()+1)
    lr = 1e-3
    input_shape = x_train.shape[1]
    nb_epoch = 50
    actv_fn = 'relu'
    batch_size = 8
    adam = Adam(lr=lr)
    total_iter = 1
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):

        # prepare the data for efficient modelling
        x_train,y_train = prepare_data(x_train,y_train)
        x_test,y_test = prepare_data(x_test,y_test)

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        # model the network
        # network : 100-100-100-9
        model = Sequential()
        model.add(Dense(100,input_dim=input_shape,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation(actv_fn))
        model.add(Dropout(0.8))

        model.add(Dense(100,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation(actv_fn))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes,
            kernel_initializer='glorot_uniform',bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        # train the network
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
        out,wt = model.fit(x_train,Y_train,batch_size=batch_size, epochs=nb_epoch, verbose=2 ,validation_data=(x_test, Y_test))
        best_wt = wt[np.argmin(out.history['val_loss'])]
        model.set_weights(best_wt)
        return model

def get_probas_clf(clf,data,label):
    data3,label3 = stratified_split3(data,label)
    total_iter = 3

    labels_list = []
    preds_list = []
    probas_list = []
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):

        # get training, testing data with data balancing
        x_train = np.concatenate((data3[n_iter%3],data3[(n_iter+1)%3]),axis=0)
        x_test = data3[(n_iter+2)%3]
        y_train = np.concatenate((label3[n_iter%3],label3[(n_iter+1)%3]),axis=0)
        y_test = label3[(n_iter+2)%3]

        # prepare the data for efficient modelling
        x_train,y_train = expand_data(x_train,y_train)
        #for i in range(9):
        #    print np.where(y_train==i)[0].size
        x_test,y_test = prepare_data(x_test,y_test)

        # model the network
        clf.fit(x_train,y_train)
        probas = clf.predict_proba(x_test)
        pred_indices = np.argmax(probas, axis=1)
        preds = np.unique(y_test)[pred_indices]

        if n_iter==0:
            labels_list = y_test.copy()
            preds_list = preds.copy()
            probas_list = probas.copy()
        else:
            labels_list = np.concatenate((labels_list,y_test.copy()),axis=0)
            preds_list = np.concatenate((preds_list,preds.copy()),axis=0)
            probas_list = np.concatenate((probas_list,probas.copy()),axis=0)
    return labels_list,preds_list,probas_list



def classifier(clf_type=None):
    if clf_type=='log_reg':
        return LogisticRegression()
    elif clf_type=='random_forest':
        return RandomForestClassifier(n_estimators=1000,
                max_depth=7, verbose=1)
    elif clf_type=='svm':
        return SVC(C=1,kernel='rbf',gamma=0.0001,probability=True)
    elif clf_type=='ann':
        return ann(data)
    elif clf_type=='xgboost':
        return XGBClassifier(max_depth=4,
                objective='multi:softprob',learning_rate=0.03333)
    else:
        return LogisticRegression()


