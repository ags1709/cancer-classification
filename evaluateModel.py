def kfold_evaluation(K, model, X, y, threshold=0.49864656):
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    all_y_true = []
    all_y_prob = []

    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    y_est_global = (np.array(all_y_prob) >= threshold).astype(int)
    cm = confusion_matrix(all_y_true, y_est_global, labels=[0, 1])

    fpr_roc, tpr_roc, thresholds = metrics.roc_curve(all_y_true, all_y_prob)
    auc = metrics.auc(fpr_roc, tpr_roc)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        
    error = (fp + fn) / len(all_y_true)
    accuracy = 1 - error
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    
    return accuracy, error, f1_score, auc, fpr, tpr, fnr, tnr

def kfold_evaluation_outliers_removed(K, model, X, y, threshold=0.49864656):
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    all_y_true = []
    all_y_prob = []

    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        
        # mean = np.mean(X_train, axis=0)
        # std = np.std(X_train, axis=0)

        non_outliers = np.all(np.abs(X_train) < 5, axis=1)

        X_train_clean = X_train[non_outliers]
        y_train_clean = y_train[non_outliers]

        model.fit(X_train_clean, y_train_clean)

        y_prob = model.predict_proba(X_test)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    y_est_global = (np.array(all_y_prob) >= threshold).astype(int)
    cm = confusion_matrix(all_y_true, y_est_global, labels=[0, 1])

    fpr_roc, tpr_roc, thresholds = metrics.roc_curve(all_y_true, all_y_prob)
    auc = metrics.auc(fpr_roc, tpr_roc)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        
    error = (fp + fn) / len(all_y_true)
    accuracy = 1 - error
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    
    return accuracy, error, f1_score, auc, fpr, tpr, fnr, tnr

def kfold_accuracy(K, model, X, y, threshold=0.49864656):
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    errors = np.zeros(K)
    i = 0
    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        fold_error = np.sum(y_pred != y_test) / len(y_test)
        errors[i] = fold_error
        i += 1

        # y_prob = model.predict_proba(X_test)[:, 1]
        
    
    
    # y_est_global = (np.array(all_y_prob) >= threshold).astype(int)
    avg_error = np.mean(errors)
        
    # error = (fp + fn) / len(all_y_true)
    accuracy = 1 - avg_error
    
    return accuracy, avg_error

def get_kfold_roc(K, model, X, y):
    CV = model_selection.KFold(n_splits=K, shuffle=True)
    
    all_y_true = []
    all_y_prob = []

    for train_index, test_index in CV.split(X):
        X_train, y_train = X[train_index, :], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]  # Get probability estimates

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    # Compute overall ROC
    fpr, tpr, thresholds = metrics.roc_curve(all_y_true, all_y_prob)
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc, thresholds

def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Random guess line
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

