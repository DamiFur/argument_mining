"""Script with functions to obtain a Pandas DataFrame with metrics.

The input is a list of dirnames and keys.
"""
import matplotlib.pyplot as plt
import numpy
import os
import pandas
import re
import seaborn as sns

from sklearn import metrics


METRIC_COLS = ['Accuracy', 'Precision', 'Recall', 'F1-Score']


def labels_single_file(filename):
    result = pandas.read_csv(filename, sep='\t')
    return result


def prediction_filenames(dirname):
    return [filename for filename in os.listdir(dirname)
            if os.path.isfile(os.path.join(dirname, filename))
               and 'predictions' in filename]


def classifier_metrics(classifier_dirpath, keys=None, averaging='macro'):
    """Reads all prediction files inside classifier_dirpath.

    Args:
        classifier_dirpath: [str] the full path to the directory with prediction
        files
        keys: [dict] map from column name to a function that receives
        the prediction filename and returns the key value. This is use in case
        there are multiple partitions of the same datasets on the directory.
        averaging: [str] type of average for precision, recall and f1-score.

    Returns:
        A DataFrame with the following columns:
        - Classifier: name of the classifier's directory
        - Dataset: 'dev' or 'test'. It's obtained from the prediction's filename
        - Accuracy
        - Precision: averaged precision
        - Recall: averaged recall
        - F1-Score: averaged f1
        - Support
        - Extra keys
    """
    column_names = ['Classifier', 'Dataset'] + METRIC_COLS + ['Support']
    if keys is not None:
        column_names.extend(keys.keys())

    result = pandas.DataFrame(columns=column_names)
    for index, prediction_file in enumerate(
            prediction_filenames(classifier_dirpath)):
        predictions = labels_single_file(
            os.path.join(classifier_dirpath, prediction_file))
        print('DEBUG: classes used {}'.format(predictions['True'].unique()))
        accuracy = metrics.accuracy_score(
            predictions['True'], predictions.Predicted)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            predictions['True'], predictions.Predicted,
            average=averaging, warn_for=[])
        support = len(predictions)
        dataset = re.search('.*_(\w+).conll', prediction_file).group(1)
        classifier = os.path.basename(os.path.normpath(classifier_dirpath))
        row = [classifier, dataset, accuracy, precision, recall, f1, support]
        if keys is not None:
            for key_name, key_function in keys.items():
                key_value = key_function(prediction_file)
                row.append(key_value)
        result.loc[index] = row
    return result


def architecture_metrics(experiment_dirs, keys=None, averaging='macro'):
    """Reads metrics for all directories in experiment_dirs.
    
    Args:
        experiment_dirs [list of str] list with the full paths to directories
        with prediction files
        keys: [dict] map from column name to a function that receives
        the prediction filename and returns the key value. This is use in case
        there are multiple partitions of the same datasets on the directory.
        averaging: [str] type of average for precision, recall and f1-score.

    Returns:
        A DataFrame with the following columns:
        - Classifier: name of the classifier's directory
        - Dataset: 'dev' or 'test'. It's obtained from the prediction's filename
        - Accuracy
        - Precision: macro precision
        - Recall: macro recall
        - F1-Score: macro f1
        - Support
        - Extra keys
    """
    result = []
    for classifier_path in experiment_dirs:
        result.append(classifier_metrics(classifier_path, keys, averaging))
    return pandas.concat(result)


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10,7),
                           fontsize=14):
    """Plots confusion matrix returned by sklearn's confusion_matrix as heatmap.
    
    Args:
        confusion_matrix: [numpy.ndarray] The numpy.ndarray object returned
            from a call to sklearn.metrics.confusion_matrix. Similarly
            constructed ndarrays can also be used.
        class_names: [list] Ordered list of class names, in the order
            they index the given confusion matrix.
        figsize: [tuple] 2-tuple, the first value determining the
            horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: [int] Font size for axes labels. Defaults to 14.
        
    Returns:
        The resulting confusion matrix figure
    """
    df_cm = pandas.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_confusion_matrix(classifier_dirpath, partition=None):
    prediction_files = prediction_filenames(classifier_dirpath)
    prediction_file = None
    for possible_pred_file in prediction_files:
        if ((partition == None or ('partition' + str(partition) in possible_pred_file))
                and 'dev' in possible_pred_file):
            prediction_file = possible_pred_file
            break
    if prediction_file is None:
        raise ValueError('No prediction for partition {}'.format(partition))
    predictions = labels_single_file(os.path.join(
        classifier_dirpath, prediction_file))
    labels = numpy.unique(numpy.concatenate(
        [predictions['True'].values, predictions.Predicted.values]))
    print(metrics.classification_report(predictions['True'],
                                        predictions.Predicted, labels=labels))
    cm = metrics.confusion_matrix(predictions['True'],
                                  predictions.Predicted, labels=labels)
    print_confusion_matrix(cm, labels)
    return None

