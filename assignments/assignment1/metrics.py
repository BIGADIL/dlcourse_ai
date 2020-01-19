def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    assert len(prediction) == len(ground_truth)
    
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    total = len(prediction)
    tp = 0
    fp = 0
    fn = 0
    for i in range(total):
        y = ground_truth[i]
        y_pred = prediction[i]
        tp += 1 if y and y_pred else 0
        fp += 1 if not y and y_pred else 0
        fn += 1 if y and not y_pred else 0
        accuracy += 1 if y == y_pred else 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = accuracy / total
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    assert len(prediction) == len(ground_truth)
    accuracy = 0
    total = len(prediction)
    for i in range(total):
        accuracy += 1 if prediction[i] == ground_truth[i] else 0
    return accuracy / total
