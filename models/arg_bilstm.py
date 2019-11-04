"""BiLSTM CNN with embeddings model derived from ukplab BiLSTM model.

It has some more flexible functions, but the core of the model is the same."""

import h5py
import json
import keras.backend as K
import logging
import math
import numpy
import tensorflow as tf

from sklearn import metrics
from ukplab_nets.neuralnets.BiLSTM import BiLSTM
from ukplab_nets.neuralnets.keraslayers.ChainCRF import (
    ChainCRF, create_custom_objects)

from keras import optimizers, layers
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


class FixedSizeBiLSTM(BiLSTM):
    """BiLSTM model with a fixed number of timesteps for sequences."""

    def __init__(self, params=None):
        super(FixedSizeBiLSTM, self).__init__(params)
        self.max_sentece_length = None  # If this is none, the longest sentence
        # in the batch is used

    def add_padded_sentences(self, train_matrix, label_name):
        labels = pad_sequences(
            [example[label_name] for example in train_matrix],
            self.max_sentece_length)
        batches = [numpy.expand_dims(labels, -1)]
        for feature_name in self.params['featureNames']:
            batches.append(pad_sequences([
                numpy.asarray(instance[feature_name])
                for instance in train_matrix], self.max_sentece_length))
        return batches

    def minibatch_iterate_dataset(self, batch_size=32, partition='trainMatrix'):
        """Create mini-batches with the same number of timesteps.

        Sentences and mini-batch chunks are shuffled and used to the
        train the model.

        Args:
            batch_size: (int) the maximum size of the partitions to create.
                The last batch may be smaller.
            partition: (str) the name of the partition to use (ex trainMatrix).
        """

        for model_name in self.modelNames:
            # Shuffle the order of the examples
            numpy.random.shuffle(self.data[model_name][partition])

        # Iterate over the examples
        batches = {}
        training_examples = len(self.data[model_name][partition])
        for start in range(0, training_examples, batch_size):
            batches.clear()
            end = start + batch_size

            for model_name in self.modelNames:
                trainMatrix = self.data[model_name][partition]
                label_name = self.labelKeys[model_name]
                batches[model_name] = self.add_padded_sentences(
                    trainMatrix[start:end], label_name)
            yield batches

    def predictLabels(self, model, sentences, batch_size=64):
        pred_labels = []
        for start in range(0, len(sentences), batch_size):
            end = start + batch_size
            instances = []
            for feature_name in self.params['featureNames']:
                input_data = pad_sequences(
                    [numpy.asarray(instance[feature_name])
                     for instance in sentences[start:end]],
                    self.max_sentece_length)
                instances.append(input_data)

            predictions = model.predict(instances, verbose=False)
            predictions = predictions.argmax(axis=-1) #Predict classes
            # We need to "unpad" the predicted labels. We use the
            # lenght of any random feature in the sentence. 
            for pred, sentence in zip(predictions, sentences[start:end]):
                sentence_len = len(sentence['tokens'])
                if sentence_len > pred.shape[0]:
                    pred_labels.append(numpy.pad(
                        pred, (0, sentence_len - pred.shape[0]), 'constant'))
                else:
                    pred_labels.append(pred[-sentence_len:])

        return pred_labels


class ArgBiLSTM(FixedSizeBiLSTM):
    """BiLSTM model tailored for argumentation mining tasks"""

    @classmethod
    def loadModel(cls, modelPath):
        model = load_model(modelPath, custom_objects=create_custom_objects())
        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            modelName = f.attrs['modelName']
            labelKey = f.attrs['labelKey']

        bilstm = cls(params)
        bilstm.setMappings(mappings, None)
        bilstm.models = {modelName: model}
        bilstm.labelKeys = {modelName: labelKey}
        bilstm.idx2Labels = {}
        bilstm.idx2Labels[modelName] = {
            v: k for k, v in bilstm.mappings[labelKey].items()}
        return bilstm

    def computeScore(self, modelName, devMatrix, testMatrix):
        return self.computeF1Scores(modelName, devMatrix, testMatrix)

    def computeF1(self, modelName, sentences):
        """Returns a traditional f1 score.

        It does not check the consistency of BIO labels."""
        labelKey = self.labelKeys[modelName]
        model = self.models[modelName]
        idx2Label = self.idx2Labels[modelName]
        true_labels = numpy.concatenate([sentences[idx][labelKey]
                         for idx in range(len(sentences))])
        pred_labels = numpy.concatenate(self.predictLabels(model, sentences))
        pre, rec, f1, _ = metrics.precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', warn_for=[])

        return pre, rec, f1

    def featuresToMerge(self):
        tokens_input = layers.Input(shape=(None,), dtype='int32',
                                    name='words_input')
        tokens = layers.Embedding(
            input_dim=self.embeddings.shape[0],
            output_dim=self.embeddings.shape[1],
            weights=[self.embeddings], trainable=False,
            mask_zero=True,
            name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue

            feature_input = layers.Input(shape=(None,), dtype='int32',
                                         name=featureName+'_input')
            feature_embedding = layers.Embedding(
                input_dim=len(self.mappings[featureName]),
                output_dim=self.params['addFeatureDimensions'],
                mask_zero=True,
                name=featureName+'_emebddings')(feature_input)

            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        return inputNodes, mergeInputLayers

    def createCharEmbeddings(self):
        # :: Character Embeddings ::
        logging.info("Pad words to uniform length for characters embeddings")
        all_sentences = []
        for dataset in self.data.values():
            for data in [dataset['trainMatrix'], dataset['devMatrix'],
                         dataset['testMatrix']]:
                for sentence in data:
                    all_sentences.append(sentence)

        self.padCharacters(all_sentences)
        logging.info("Words padded to %d characters" % (self.maxCharLen))

        charset = self.mappings['characters']
        charEmbeddingsSize = self.params['charEmbeddingsSize']
        maxCharLen = self.maxCharLen
        charEmbeddings= []
        for _ in charset:
            limit = math.sqrt(3.0/charEmbeddingsSize)
            vector = numpy.random.uniform(-limit, limit, charEmbeddingsSize)
            charEmbeddings.append(vector)

        charEmbeddings[0] = numpy.zeros(charEmbeddingsSize) #Zero padding
        return numpy.asarray(charEmbeddings)

    def addCharInput(self):
        return layers.Input(shape=(None, self.maxCharLen), dtype='int32',
                            name='char_input')

    def addCharEmbeddingLayers(self, inputNodes, mergeInputLayers,
                               chars_input, charEmbeddings):
        # Use LSTM for char embeddings from Lample et al., 2016
        if self.params['charEmbeddings'].lower() == 'lstm':
            chars = layers.TimeDistributed(
                layers.Embedding(
                    input_dim=charEmbeddings.shape[0],
                    output_dim=charEmbeddings.shape[1],
                    weights=[charEmbeddings], trainable=True, mask_zero=True),
                name='char_emd')(chars_input)
            charLSTMSize = self.params['charLSTMSize']
            chars = layers.TimeDistributed(
                layers.Bidirectional(
                    layers.LSTM(charLSTMSize, return_sequences=False)),
                name="char_lstm")(chars)
        else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
            chars = layers.TimeDistributed(
                layers.Embedding(  # Conv layer does not support masking
                    input_dim=charEmbeddings.shape[0],
                    output_dim=charEmbeddings.shape[1], trainable=True,
                    weights=[charEmbeddings]), name='char_emd')(chars_input)
            charFilterSize = self.params['charFilterSize']
            charFilterLength = self.params['charFilterLength']
            chars = layers.TimeDistributed(
                layers.Conv1D(charFilterSize, charFilterLength, padding='same'),
                name="char_cnn")(chars)
            chars = layers.TimeDistributed(layers.GlobalMaxPooling1D(),
                                           name="char_pooling")(chars)

        mergeInputLayers.append(chars)
        inputNodes.append(chars_input)
        self.params['featureNames'].append('characters')

    def handleTasks(self, mergeInputLayers, inputNodes):
        # :: Task Identifier ::
        if self.params['useTaskIdentifier']:
            self.addTaskIdentifier()

            taskID_input = layers.Input(shape=(None,), dtype='int32',
                                        name='task_id_input')
            taskIDMatrix = numpy.identity(len(self.modelNames), dtype='float32')
            taskID_outputlayer = layers.Embedding(
                input_dim=taskIDMatrix.shape[0],
                output_dim=taskIDMatrix.shape[1], weights=[taskIDMatrix],
                trainable=False, name='task_id_embedding')(taskID_input)

            mergeInputLayers.append(taskID_outputlayer)
            inputNodes.append(taskID_input)
            self.params['featureNames'].append('taskID')

    def addRecurrentLayers(self, merged_input):
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        for count, size in enumerate(self.params['LSTM-Size']):
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = layers.Bidirectional(
                    layers.LSTM(size, return_sequences=True,
                                dropout=self.params['dropout'][0],
                                recurrent_dropout=self.params['dropout'][1]),
                    name='shared_varLSTM_' + str(count))(shared_layer)
            else:
                # Naive dropout
                shared_layer = layers.Bidirectional(
                    layers.LSTM(size, return_sequences=True),
                    name='shared_LSTM_'+str(count))(shared_layer)
                if self.params['dropout'] > 0.0:
                    layer_name = ('shared_dropout_' + str(
                                  self.params['dropout']) + "_" + str(count))
                    shared_layer = layers.TimeDistributed(
                        layers.Dropout(self.params['dropout']),
                        name=layer_name)(shared_layer)
        return shared_layer

    def addPreAttentionLayer(self, merged_input):
        """Add attention mechanisms to the tensor merged_input.

        Args:
            merged_input: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the concatenation of features.
        """
        return merged_input

    def addPostAttentionLayer(self, shared_layer):
        """Add attention mechanisms to the tensor shared_layer.

        Args:
            shared_layer: 3-dimensional Tensor, where the first
            dimension corresponds to the batch size, the second to the sequence
            timesteps and the last one to the output of the BiLSTM.
        """
        return shared_layer

    def addOutput(self, modelName, shared_layer):
        output = shared_layer

        if modelName in self.params['customClassifier']:
            modelClassifier = self.params['customClassifier'][modelName]
        else:
            modelClassifier = self.params['classifier']

        if not isinstance(modelClassifier, (tuple, list)):
            modelClassifier = [modelClassifier]

        cnt = 1
        for classifier in modelClassifier:
            n_class_labels = len(self.mappings[self.labelKeys[modelName]])

            if classifier == 'Softmax':
                output = layers.TimeDistributed(
                    layers.Dense(n_class_labels, activation='softmax'),
                    name=modelName+'_softmax')(output)
                lossFct = 'sparse_categorical_crossentropy'
            elif classifier == 'CRF':
                output = layers.TimeDistributed(
                    layers.Dense(n_class_labels, activation=None),
                    name=modelName + '_hidden_lin_layer')(output)
                crf = ChainCRF(name=modelName+'_crf')
                output = crf(output)
                lossFct = crf.sparse_loss
            elif (isinstance(classifier, (list, tuple)) and
                  classifier[0] == 'LSTM'):
                size = classifier[1]
                if isinstance(self.params['dropout'], (list, tuple)):
                    output = layers.Bidirectional(
                        layers.LSTM(size, return_sequences=True,
                             dropout=self.params['dropout'][0],
                             recurrent_dropout=self.params['dropout'][1]),
                        name=modelName+'_varLSTM_'+str(cnt))(output)
                else:
                    """ Naive dropout """
                    output = layers.Bidirectional(
                        layers.LSTM(size, return_sequences=True),
                        name=modelName+'_LSTM_'+str(cnt))(output)
                    if self.params['dropout'] > 0.0:
                        output = layers.TimeDistributed(
                            layers.Dropout(self.params['dropout']),
                            name=modelName+'_dropout_'+str(
                                self.params['dropout'])+"_"+str(cnt))(output)
            else:
                assert(False)  # Wrong classifier
            cnt += 1
        return output, lossFct

    def getOptimizer(self):
        # :: Parameters for the optimizer ::
        optimizerParams = {}
        if ('clipnorm' in self.params and self.params['clipnorm'] != None and
                self.params['clipnorm'] > 0):
            optimizerParams['clipnorm'] = self.params['clipnorm']
        if ('clipvalue' in self.params and self.params['clipvalue'] != None and
                self.params['clipvalue'] > 0):
            optimizerParams['clipvalue'] = self.params['clipvalue']

        if self.params['optimizer'].lower() == 'adam':
            opt = optimizers.Adam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = optimizers.Nadam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'rmsprop':
            opt = optimizers.RMSprop(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = optimizers.Adadelta(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = optimizers.Adagrad(**optimizerParams)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = optimizers.SGD(lr=0.1, **optimizerParams)
        return opt

    def buildModel(self):
        # Refactoring code so it's easier to add attention. Same functionalities
        # as original BiLSTM network.
        self.models = {}

        # TensorFlow wizardry
        config = tf.compat.v1.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))

        inputNodes, mergeInputLayers = self.featuresToMerge()
        if self.params['charEmbeddings'] not in [
                None, "None", "none", False, "False", "false"]:
            char_embeddings = self.createCharEmbeddings()
            char_input = self.addCharInput()
            self.addCharEmbeddingLayers(inputNodes, mergeInputLayers,
                                        char_input, char_embeddings)

        self.handleTasks(inputNodes, mergeInputLayers)

        if len(mergeInputLayers) >= 2:
            merged_input = layers.concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]

        merged_input = self.addPreAttentionLayer(merged_input)
        shared_layer = self.addRecurrentLayers(merged_input)
        # TODO improve this beacuse it makes no sense to add the attention to
        # the shared task.
        shared_layer = self.addPostAttentionLayer(shared_layer)

        for modelName in self.modelNames:
            output, loss_function = self.addOutput(modelName, shared_layer)
            optimizer = self.getOptimizer()

            model = Model(inputs=inputNodes, outputs=[output])
            model.compile(loss=loss_function, optimizer=optimizer)

            model.summary(line_length=200)
            #logging.info(model.get_config())
            #logging.info("Optimizer: %s - %s" % (str(type(model.optimizer)), str(model.optimizer.get_config())))

            self.models[modelName] = model
