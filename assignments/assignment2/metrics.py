def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    assert len(prediction) == len(ground_truth)
    accuracy = 0
    total = len(prediction)
    for i in range(total):
        accuracy += 1 if prediction[i] == ground_truth[i] else 0
    return accuracy / total