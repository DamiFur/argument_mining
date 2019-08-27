"""Script to train a BiLSTM-CNN-CRF model from UKPLab repository

To run the script, clone the repository
https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf.git
under the name ukplab_nets and add it the path to PYTHONPATH.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report


MAX_LEN = 75
bs = 32

# import argparse
# import logging
# import os
# import sys
# parent = os.path.abspath('..')
# sys.path.insert(0, parent)
# import utils



# loggingLevel = logging.INFO
# logger = logging.getLogger()
# logger.setLevel(loggingLevel)

# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(loggingLevel)
# formatter = logging.Formatter('%(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


# def read_args():
#     parser = argparse.ArgumentParser(
#         description='Training a bi-directional RNN')
#     # Classifier parameters
#     parser.add_argument('--num_units', nargs='+', default=[100, 100], type=int,
#                         help='Number of hidden units in RNN')
#     parser.add_argument('--dropout', nargs='+', default=[0.5, 0.5], type=float,
#                         help='Dropout ratio for every layer')
#     parser.add_argument('--char_embedding', type=str, default=None,
#                         choices=['None', 'lstm', 'cnn'],
#                         help='Type of character embedding. Options are: None, '
#                         'lstm or cnn. LSTM embeddings are from '
#                         'Lample et al., 2016, CNN embeddings from '
#                         'Ma and Hovy, 2016')
#     parser.add_argument('--char_embedding_size', type=int, default=30,
#                         help='Size of the character embedding. Use 0 '
#                         'for no embedding')
#     # Training parameters
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Number of sentences in each batch')
#     parser.add_argument('--patience', type=int, default=5,
#                         help='Number of iterations of lower results before '
#                         'early stopping.')
#     # TODO add options for char embedding sizes
#     # TODO add options for clipvalue and clipnorm

#     # Pipeline parametres
#     parser.add_argument('--dataset', type=str,
#                         help='Path to the pickled file with the dataset')
#     parser.add_argument('--output_dirpath', type=str,
#                         help='Path to store the performance scores for dev '
#                              'and test datasets')
#     parser.add_argument('--epochs', type=int, default=100,
#                         help='Number of epochs to train the classifier')
#     parser.add_argument('--classifier', type=str, default='CRF',
#                         help='Classifier type (last layer). Options are '
#                              'CRF or Softmax.')
#     parser.add_argument('--experiment_name', type=str, default=None,
#                         help='Name of the experiment to store the results')
#     parser.add_argument('--attention_model', type=str, default='None',
#                         help='Use the specified attention mechanism. Options: '
#                              'None, ' + ', '.join(ATTENTION_MODELS.keys()))
#     parser.add_argument('--attention_activation', type=str, default=None,
#                         help='Use the specified attention activation. Options: '
#                              'tanh, sigmoid')
#     args = parser.parse_args()

#     assert len(args.num_units) == len(args.dropout)
#     return args


# def load_dataset(filename):
#     pickled_object = utils.pickle_from_file(filename)
#     return (pickled_object['embeddings'], pickled_object['mappings'],
#             pickled_object['data'], pickled_object['datasets'])

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["text"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def main():
    """Training pipeline"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()


    data = pd.read_csv("train_m.txt", sep='\t', encoding="latin1").fillna(method="ffill")

    getter = SentenceGetter(data)

    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    labels = [[s[1] for s in sent] for sent in getter.sentences]

    tags_vals = list(set(data["tag"].values))
    tag2idx = {t: i for i, t in enumerate(tags_vals)}

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")

    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
    
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

    model.cuda();

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    epochs = 5
    max_grad_norm = 1.0

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print("Precision-Score: {}".format(precision_score(pred_tags, valid_tags)))
        print("Recall-Score: {}".format(recall_score(pred_tags, valid_tags)))
        print(classification_report(pred_tags, valid_tags))
        

    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
                
        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("Validation loss: {}".format(eval_loss/nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print("Precision-Score: {}".format(precision_score(pred_tags, valid_tags)))
    print("Recall-Score: {}".format(recall_score(pred_tags, valid_tags)))
    print(classification_report(pred_tags, valid_tags))

    true_positives_O = 0
    predicted_positives_O = 0
    real_positives_O = 0
    for pred, valid in zip(pred_tags, valid_tags):
        if pred == 'I-claim' and valid == 'I-claim':
            true_positives_O += 1
        if pred == 'I-claim':
            predicted_positives_O += 1
        if valid == 'I-claim':
            real_positives_O += 1

    print("True positives I: {}".format(true_positives_O))
    print("predicted positives I: {}".format(predicted_positives_O))
    print("real positives I: {}".format(real_positives_O))
    
    true_positives_B = 0
    predicted_positives_B = 0
    real_positives_B = 0
    for pred, valid in zip(pred_tags, valid_tags):
        if pred == 'B-claim' and valid == 'B-claim':
            true_positives_B += 1
        if pred == 'B-claim':
            predicted_positives_B += 1
        if valid == 'B-claim':
            real_positives_B += 1
    
    print("True positives B: {}".format(true_positives_B))
    print("predicted positives B: {}".format(predicted_positives_B))
    print("real positives B: {}".format(real_positives_B))
    
    with open("resultados", 'w') as out:
        out.write("Predictions:\n")
        out.write("{}".format(list(zip(list(val_inputs), pred_tags, valid_tags))))


if __name__ == '__main__':
    main()
