## Evaluation
---

The evaluation of ML models is non trivial, and it is not sufficient to
compare the top results of different classifiers. The objective of this project
is to assess the impact of attention mechanisms on argument mining tasks, which
requires an analysis from different angles. We will focus on:

- **Quantitative analysis**: we calculate and compare classifier performance
with standard classification metrics. We have selected Accuracy, Precision,
Recall and F1-Score, the last three with macro average unless the dataset requires a
detailed analysis. The primary metric is the F1-Score.

As stated by Reimers et al. (2017), the quantitative analysis must include an
estimation of the distribution of classifiers with respect to: different random
initialization of parameters, and different selection of hyperparameters.

We will use the performance results not only to assess which is the best
architecture for each task, but also to obtain insights on the task itself, by
analyzing which are the most adequate hyperparameters. For example, the value of
the hyperparameter *maximum sequence length processed at a time* gives an
estimation of the range of relations between elements of a sequence, versus
the capability of the model to learn larger ranges.

- **Qualitative analysis**: one of the main advantages of attention models is
they bring interpretability to the model, which must not be confused with
explainability.

Moreover, in this project with have different models (without attention,
with word and context attention, with self-attention and with bert model
language) applied over different datasets, and different tasks (claim-detection,
argumentative component classification). As a result, for a consistent
evaluation we need to combine different formats of data, coming from
different servers.

## Requirements

* Ensure the classifiers compared (by task and by dataset)
belong to the same type of experiment. This includes, same or comparable
test datasets, same pre-process steps, same word embeddings.

* Use the same metrics for all experiments. We have selected the sklearn
implementation of metrics with a macro average, when applicable, calculated
over all words on the test dataset. If the same architecture
(combination of hyperparameters) is trained more than once, the average of all
runs is reported.

* Produce legible and adequate visualizations with the same style and
color palette for all experiments.

## Conventions

#### General aspect

Import the file `visualizations.py`, this will set the same seaborn context
for all notebooks. Make graphics with high definition, for example with
height/size 8.

#### Naming

1. Datasets: echr, essays, abstracts, ibm

2. Tasks: component (component classification), claim (claim detection)

3. Attention types:
  a. In visualizations and tables: No Attention, Word Attention,
     Context Attention, Self Attention, Bert
  b. In filenames: none, word, context, self, bert

4. Activations: (in all) none, linear, sigmoid, tanh, relu

#### Colors

Use the corresponding palettes defined in `visualizations.py`. If the plot
has a single color, use colors.blue.

## Codebase organization

The platform selected for the evaluation are Jupyter notebooks for two main
reasons: they allow visualizations on remote servers, and

So far, we have the following steps of evaluation:

1. Calculating individual metrics over classifiers. For this, we use individual
notebooks like `metrics_echr_claim_detection`. Because
each server and task has different paths to results, each must have
different notebooks.
  a. ACTION ITEM: Remove common code into script [Done]
  b. ACTION ITEM: Rename all notebooks according to convention:
    `metrics_[dataset]_[task]`
  c. ACTION ITEM: Remove old notebooks from repository
  d. ACTION_ITEM: Clean code in notebooks and use functions from scripts.

2. Obtaining distribution of results. For each task (claim detection and
component classification) and dataset, we need to obtain the following
visualizations:
  1. Comparing attention mechanisms. We obtain one boxplot/boxenplot
  comparing all F1 scores obtained divided by the type of
  attention (none, word, context, self, bert). This will probably be included
  in publications. As a result, we need to condense the information without
  making an overwhelmingly large graphic.
  2. Comparing hyperparameters. We obtain one boxplot/boxenplot/swarmplot for
  the distribution of F1 scores obtained when varying the particular
  hyperparameter. This is only for exploration, only the best visualizations
  will be included in publications.
