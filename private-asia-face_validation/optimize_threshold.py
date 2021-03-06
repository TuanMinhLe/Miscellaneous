import numpy as np
from sklearn.model_selection import KFold

"""
    Find optimal threshold for ROC curve
"""


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds):
    assert embeddings1.shape[0] == embeddings2.shape[0]
    assert embeddings1.shape[1] == embeddings2.shape[1]
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))

    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # Squared L2 distance
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    # Find the best threshold for the fold
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set]
            )
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            (
                tprs[fold_idx, threshold_idx],
                fprs[fold_idx, threshold_idx],
                _,
            ) = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set]
        )

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds,
    )

    return tpr, fpr, accuracy, best_thresholds
