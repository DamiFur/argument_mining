"""Script to train a BiLSTM-CNN-CRF model from UKPLab repository

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.
"""

import argparse
import logging
import numpy
import os
import pandas
import sys
parent = os.path.abspath('..')
sys.path.insert(0, parent)
import utils
from models.selfatt_arg_bilstm import SelfAttArgBiLSTM
from sklearn import metrics


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
    parser.add_argument('--n_heads', type=int,
                        help='Number of attention heads.')
    parser.add_argument('--attention_size', type=int,
                        help='Size of the attention layer\'s output.')
    parser.add_argument('--target_column', type=str, default='arg_component',
                        help='Name of the column to use as label.')
    args = parser.parse_args()

    assert len(args.num_units) == len(args.dropout)
    return args


def load_dataset(filename):
    pickled_object = utils.pickle_from_file(filename)
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
        'n_heads':  args.n_heads,  'attention_size': args.attention_size
    }
    print(classifier_params)

    print('Attention model: self')
    model = SelfAttArgBiLSTM(classifier_params)
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

    dataset_name = [x for x in data.keys()][0]  # I hate python 3
    label_encoding = {value: key
                      for key, value in mappings[args.target_column].items()}

    def tag_dataset(partition_name):
        partition_name_short = 'dev' if 'dev' in partition_name else 'test'
        output_filename = os.path.join(
            args.output_dirpath, 'predictions_{}_{}_{}.conll'.format(
                args.experiment_name, dataset_name, partition_name_short))

        tags = model.tagSentences(data[dataset_name][partition_name])
        true_labels = []
        result = []

        for idx, (sentence, sentence_labels) in enumerate(zip(
                data[dataset_name][partition_name], tags[dataset_name])):
            for token, true_label_id, predicted_label in zip(
                    sentence['raw_tokens'], sentence[args.target_column],
                    sentence_labels):
                if token == 'PADDING_TOKEN':
                    continue
                true_label = label_encoding[true_label_id]
                true_labels.append(true_label)
                result.append((token, true_label, predicted_label, idx))

        result = pandas.DataFrame(
            result, columns=['Token', 'True', 'Predicted', 'Sentence'])
        result.to_csv(output_filename, sep='\t', index=False)
        print(metrics.classification_report(
    	    true_labels, numpy.concatenate(tags[dataset_name])))

    tag_dataset('devMatrix')
    tag_dataset('testMatrix')


if __name__ == '__main__':
    main()
