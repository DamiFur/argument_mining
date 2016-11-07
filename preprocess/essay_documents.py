"""Abstractions to handle EssayDocuments"""

import six

from nltk import pos_tag as pos_tagger
from nltk.tokenize import sent_tokenize, word_tokenize


class Sentence(object):
    """Abstraction of a sentence."""
    def __init__(self, sentence_position, default_label='O'):
        # Position of the sentence in the document
        self.position = sentence_position
        self.words = []
        # Saves the start position in the original document for each word.
        self.word_positions = []
        self.pos = []
        self.labels = []
        self.tree = None
        self.default_label = default_label
        # A map from start positions (in document) to word index in self.words
        self.word_index = {}

    def add_word(self, word, pos, position, label=None):
        if not label:
            label = self.default_label
        self.words.append(word)
        self.pos.append(pos)
        self.labels.append(label)
        self.word_positions.append(position)
        self.word_index[position] = len(self.words) - 1

    def build_from_text(self, text, initial_position=0):
        raw_words = word_tokenize(text)
        pos_tags = pos_tagger(raw_words)
        last_position = 0
        for word, pos in pos_tags:
            last_position = text[last_position:].find(word) + last_position
            assert last_position >= 0
            self.add_word(word, pos, last_position + initial_position,
                          self.default_label)

    @property
    def start_position(self):
        if not self.words:
            return -1
        return self.word_positions[0]

    @property
    def end_position(self):
        if not self.words:
            return -1
        return self.word_positions[-1] + len(self.words[-1])

    def get_word_for_position(self, position):
        return self.word_index.get(position, -1)

    def add_label_for_position(self, label, start, end):
        """Adds label to words with positions between start and end (including).

        Returns the last position of char_range used. If char_range doesn't
        intersect with sentence, then does nothing."""
        if end < self.start_position or start > self.end_position:
            return start
        while start <= end and start < self.end_position:
            word_index = self.get_word_for_position(start)
            if word_index < 0:
                start += 1
                continue
            self.labels[word_index] = label
            start += len(self.words[word_index])
        return start


class EssayDocument(object):
    def __init__(self, identifier, default_label='O', title=''):
        # List of Sentences.
        self.sentences = []
        # Representation of document as continous text.
        self.text = ''
        self.identifier = identifier
        self.default_label = default_label
        self.title = title
        self.parse_trees = []

    def build_from_text(self, text, start_index=0):
        self.text = text
        raw_sentences = sent_tokenize(six.text_type(text))
        for index, raw_sentence in enumerate(raw_sentences):
            sentence = Sentence(index)
            initial_position = text.find(raw_sentence) + start_index
            assert initial_position >= 0
            sentence.build_from_text(raw_sentence, initial_position)
            self.sentences.append(sentence)

    def add_label_for_position(self, label, start, end):
        """Adds the given label to all words covering range."""
        assert start >= 0 and end <= self.sentences[-1].end_position
        last_start = start
        for sentence in self.sentences:
            last_start = sentence.add_label_for_position(label, last_start, end)
            if last_start == end:
                break

    def parse_text(self, parser):
        for sentence in self.sentences:
            self.parse_trees.append(parser.parse(sentence.words).next())