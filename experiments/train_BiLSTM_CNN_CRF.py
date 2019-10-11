"""Script to train a BiLSTM-CNN-CRF model from UKPLab repository

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.
"""

import argparse
import logging
import os
import sys
parent = os.path.abspath('..')
sys.path.insert(0, parent)
from utils import pickle_from_file
from models.arg_bilstm import ArgBiLSTM
from models.att_arg_bilstm import TimePreAttArgBiLSTM, FeaturePreAttArgBiLSTM
import tensorflow as tf

ATTENTION_MODELS = {
    'time_pre': TimePreAttArgBiLSTM,
    'feature_pre': FeaturePreAttArgBiLSTM,
}


loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a bi-directional RNN')
    # Classifier parameters
    parser.add_argument('--num_units', nargs='+', default=[100, 100], type=int,
                        help='Number of hidden units in RNN')
    parser.add_argument('--dropout', nargs='+', default=[0.5, 0.5], type=float,
                        help='Dropout ratio for every layer')
    parser.add_argument('--char_embedding', type=str, default=None,
                        choices=['None', 'lstm', 'cnn'],
                        help='Type of character embedding. Options are: None, '
                        'lstm or cnn. LSTM embeddings are from '
                        'Lample et al., 2016, CNN embeddings from '
                        'Ma and Hovy, 2016')
    parser.add_argument('--char_embedding_size', type=int, default=30,
                        help='Size of the character embedding. Use 0 '
                        'for no embedding')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of sentences in each batch')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of iterations of lower results before '
                        'early stopping.')
    # TODO add options for char embedding sizes
    # TODO add options for clipvalue and clipnorm

    # Pipeline parametres
    parser.add_argument('--dataset', type=str,
                        help='Path to the pickled file with the dataset')
    parser.add_argument('--output_dirpath', type=str,
                        help='Path to store the performance scores for dev '
                             'and test datasets')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train the classifier')
    parser.add_argument('--classifier', type=str, default='CRF',
                        help='Classifier type (last layer). Options are '
                             'CRF or Softmax.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment to store the results')
    parser.add_argument('--attention_model', type=str, default='None',
                        help='Use the specified attention mechanism. Options: '
                             'None, ' + ', '.join(ATTENTION_MODELS.keys()))
    parser.add_argument('--attention_activation', type=str, default=None,
                        help='Use the specified attention activation. Options: '
                             'tanh, sigmoid')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset(filename):
    pickled_object = pickle_from_file(filename)
    return (pickled_object['embeddings'], pickled_object['mappings'],
            pickled_object['data'], pickled_object['datasets'])


def main():
    """Training pipeline"""
    args = read_args()

    # Read dataset
    embeddings, mappings, data, datasets = load_dataset(args.dataset)

    classifier_params = {
        'classifier': args.classifier, 'LSTM-Size': args.num_units,
        'dropout': args.dropout, 'charEmbeddingsSize': args.char_embedding_size,
        'charEmbeddings': args.char_embedding, 'miniBatchSize': args.batch_size,
        'earlyStopping': args.patience,
        'attentionActivation': args.attention_activation,
    }
    print(classifier_params)

    attention_model = ATTENTION_MODELS.get(args.attention_model, None)
    print('Attention model: {}'.format(attention_model))
    if attention_model is None:
        model = ArgBiLSTM(classifier_params)
    else:
        model = attention_model(classifier_params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets, data)
    # Path to store performance scores for dev / test
    if args.experiment_name is None:
        results_filename = os.path.join(
            args.output_dirpath,
            '_'.join([args.classifier, str(args.char_embedding)] +
                     [str(x) for x in args.num_units])
        )
    else:
        results_filename = os.path.join(args.output_dirpath,
                                        args.experiment_name + "_results.txt")
    model.storeResults(results_filename)
    # Path to store models. We only want to store the best model found until
    # the moment
    model.modelSavePath = os.path.join(
        args.output_dirpath, "{}_model.h5".format(args.experiment_name))
    model.fit(epochs=args.epochs)


if __name__ == '__main__':
    main()
