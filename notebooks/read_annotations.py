"""Functions to read annotations and convert them in AnnotatedDocument instances"""

import itertools
import os
import re
import sys

from collections import defaultdict

def append_path(module_path):
    if module_path not in sys.path:
        sys.path.append(module_path)

append_path(os.path.abspath('..'))

from preprocess import annotated_documents, arg_docs2conll

ANNOTATIONS_DIR = '/home/milagro/am/third_party/brat-v1.3_Crunchy_Frog/data/'
ANNOTATORS = {
    'mili': {'dirname': 'judgements-mili'},
    'laura': {'dirname': 'judgements-laura'},
    # 'serena': {'dirname': 'judgements-serena'},
    'cristian': {'dirname': 'judgements-cristian'}
}
ANNOTATION_FORMAT = r'.*\.ann'
BRAT_DIRNAME = '/home/milagro/FaMAF/am/third_party/brat/'


# Find files to compare
def get_non_empty_filenames(input_dirpath, pattern, size_limit=500):
    """Returns the names of the files in input_dirpath matching pattern."""
    all_files = os.listdir(input_dirpath)
    result = {}
    for filename in all_files:
        if not re.match(pattern, filename):
            continue
        filepath = os.path.join(input_dirpath, filename)
        if os.path.isfile(filepath) and os.stat(filepath).st_size > 500:
            result[filename] = filepath
    return result


def get_filenames_by_annotator():
    files = defaultdict(lambda: {})
    for name, annotator in ANNOTATORS.items():
        annotator['files'] = get_non_empty_filenames(
            os.path.join(ANNOTATIONS_DIR, annotator['dirname']),
            ANNOTATION_FORMAT)
        for filename, filepath in annotator['files'].items():
            files[filename][name] = filepath
    return files


def get_filenames():
    filenames = {}
    for name, annotator in ANNOTATORS.items():
        for filename in get_non_empty_filenames(
                os.path.join(ANNOTATIONS_DIR, annotator['dirname']),
                ANNOTATION_FORMAT).values():
            filenames[name] = filename
    return filenames


def get_annotated_documents():
    files = get_filenames_by_annotator()
    document_pairs = []
    for value in files.values():
        if len(value) < 2:
            continue
        annotations = read_annotations(value.items())
        for ann1, ann2 in list(itertools.combinations(annotations.keys(), 2)):
            document_pairs.append((annotations[ann1], annotations[ann2]))
    return document_pairs, annotations


def read_annotations(filenames):
    annotations = {}
    for name, filename in filenames:
        identifier = 'Case: {} - Ann: {}'.format(
            os.path.basename(filename[:-4]).replace(
                'CASE_OF__', '').replace('_', ' '),
            name[0].title())
        with arg_docs2conll.AnnotatedDocumentFactory(
                filename.replace('ann', 'txt'), identifier) as instance_extractor:
            annotations[name] = instance_extractor.build_document()
    return annotations


def get_labels(doc1, doc2):
    words1, labels1 = doc1.get_word_label_list()
    words2, labels2 = doc2.get_word_label_list()
    # Check the documents are equal
    assert words1 == words2
    return labels1, labels2
