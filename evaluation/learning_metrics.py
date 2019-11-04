"""Script with functions to read the learning metrics from classifier outputs.
"""

import pandas
import os
import re


def learning_single_file(filename):
    result = pandas.read_csv(
        filename, sep='\t', header=None,
        names=['epoch', 'model_name', 'dev_score', 'test_score',
               'max_dev_score', 'max_test_score'])
    return result


def learning_filenames(dirname):
    return [filename for filename in os.listdir(dirname)
            if os.path.isfile(os.path.join(dirname, filename))
                and 'results' in filename]


def classifier_learning(classifier_dirpath, keys=None):
    """Reads all learning files in classifier dirpath.

    Args:
        classifier_dirpath: [str] the full path to the directory with prediction
        files.
        keys: [dict] map from column name to a function that receives
        the prediction filename and returns the key value. This is use in case
        there are multiple partitions of the same datasets on the directory.

    Returns:
        A DataFrame with the following columns:
        - epoch: the epoch number
        - Metric Value
        - Dataset: test or dev
        - Classifier: name of the classifier's directory
        - Extra keys
    """
    result = []
    for index, result_file in enumerate(learning_filenames(classifier_dirpath)):
        learning_metrics = learning_single_file(
            os.path.join(classifier_dirpath, result_file)).drop(
            columns=['model_name', 'max_dev_score', 'max_test_score'])
        learning_metrics = learning_metrics.set_index(
            ['epoch']).stack().reset_index().rename(
            columns={0: 'Metric Value', 'level_1': 'Dataset'})
        learning_metrics['Dataset'] = learning_metrics['Dataset'].replace(
            to_replace='dev_score', value='dev').replace(
            to_replace='test_score', value='test')
        learning_metrics['Classifier'] = os.path.basename(
            os.path.normpath(classifier_dirpath))
        if keys is not None:
            for key_name, key_function in keys.items():
                learning_metrics.loc[:, key_name] = key_function(result_file)
        result.append(learning_metrics)
    return pandas.concat(result)


def architecture_learning(experiment_dirs, keys=None):
    """Reads learning metrics for all classifiers

    Args:                                                                        
         classifier_dirpath: [str] the full path to the directory with prediction 
         files.                                                                   
         keys: [dict] map from column name to a function that receives            
         the prediction filename and returns the key value. This is used when   
         there are multiple partitions of the same datasets on the directory.

    Returns:
        A DataFrame with the following columns:
        - epoch: the epoch number
        - Metric Value
        - Dataset: test or dev
        - Classifier: name of the classifier's directory
        - Extra keys
    """
    result = []
    for classifier_path in experiment_dirs:
        result.append(classifier_learning(classifier_path, keys))
    return pandas.concat(result)
