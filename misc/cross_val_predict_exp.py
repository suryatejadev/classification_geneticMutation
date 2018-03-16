def cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict'):

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]



