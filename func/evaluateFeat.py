from modules import *
from classifiers import *

#The function cross_val_predict has a similar interface to cross_val_score, but returns, for each element in the input, the prediction that was obtained for that element when it was in the test set. Only cross-validation strategies that assign all elements to a test set exactly once can be used (otherwise, an exception is raised).

def evaluate_features(X, y, X_test, folder_name,clf_name=None):
    """General helper function for evaluating effectiveness of passed features in ML model
    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation
    Args:
        X (array-like): Features array. Shape (n_samples, n_features)
        y (array-like): Labels array. Shape (n_samples,)
        clf: Classifier to use. If None, default Log reg is use.
    StratifiedKFold splits the data into train and test, while preserving the
    ratio of the classes. Default is 3 fold. So, the outputs of the validation
    data during the 3 folds are concatenated and returned as output
    """
    if clf_name=='ann':
        probas = ann(X,y)
    else:
        clf = classifier(clf_name)
        if clf is None:
            clf = LogisticRegression()
        probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(random_state=8),
                                  n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    plot_confusion_matrix(y, preds,normalize=True)
    plt.savefig('outputs/'+folder_name+'/new_res/CM_100_'+clf_name+'.jpg')

    '''clf = clf.fit(X,y)
    test_proba = clf.predict_proba(X_test[1])
    test_pred = np.zeros_like(test_proba)
    test_pred[np.arange(len(test_proba)), test_proba.argmax(1)] = 1
    test_output = np.column_stack((X_test[0],test_pred)).astype(int)
    header = 'ID,class1,class2,class3,class4,class5,class6,class7,class8,class9'
    np.savetxt('outputs/'+folder_name+'/test_prediction_100_'+clf_name+'.csv',
            test_output,fmt='%i',delimiter=',',header=header,comments='')'''

    return log_loss(y, probas),accuracy_score(y, preds)

