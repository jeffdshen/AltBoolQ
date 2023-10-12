# AltBoolQ

## Introduction

This repository contains the code for the [Stanford CS229: Machine Learning](https://cs229.stanford.edu/) final project titled "Can Pretrained Language Models Understand Without Using Prior World Knowledge?", which won a CS229 Best Project Award in [Spring 2022](https://cs229.stanford.edu/syllabus-spring2022.html). The final report can be found [here](./CS229__Project_Final_Report.pdf).

The project introduces a method for randomizing entities in examples for QA tasks so that models cannot solely rely on world knowledge to answer questions. **AltBoolQ** is the resulting dataset from applying the method to the [BoolQ](https://github.com/google-research-datasets/boolean-questions) dataset. Several findings from the project include:
1. AltBoolQ is more difficult than BoolQ, and mixing in training on AltBoolQ can boost performance on the BoolQ dataset.
2. BoolQ may include statistical cues that make it too easy. A possible diagnostic is to mask out relevant entities in text, which should drop model performance to the majority class if there are no statistical regularities in the language. There is *surprisingly* little drop in model performance on BoolQ.
3. Models finetuned on only BoolQ transfer well to AltBoolQ with *no additional training*. This robustness is surprising and suggests models are relying not on world knowledge, but on language understanding or statistical cues.

## Results

We also construct a dataset based on BoolQ with questions only (i.e. closed-book), which we term **QBoolQ**, and a dataset with selected entities redacted, which we term **MaskedBoolQ**. Here are the accuracies for finetuned [BERT](https://github.com/google-research/bert) and [RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) models on these datasets:

Model | BoolQ | QBoolQ | MaskedBoolQ | AltBoolQ
--- | :---: | :---: |:---: | :---: 
BERT-base     | 74.31 | 66.06 | 74.93 | 72.38
RoBERTa-base  | 81.01 | 66.33 | 77.79 | 76.21
BERT-large    | 77.65 | 66.57 | 76.89 | 76.10
RoBERTa-large | 86.18 | 67.31 | 82.09 | 82.92

The majority class is 62.74% of each dataset. The good performance on MaskedBoolQ is very surprising because over 20% of words are masked with the word "redacted" in MaskedBoolQ, suggesting BoolQ contains statistical cues. Here are models trained on BoolQ, AltBoolQ, or both, and their accuracies evaluated on the other datasets:

Model | Train set | BoolQ | AltBoolQ | BoolQ + AltBoolQ
--- | --- | :---: |:---: | :---: 
BERT-base | BoolQ | 74.31 | 71.25 | 72.87 
BERT-base | AltBoolQ | 71.01 | 72.38 | 71.65 
BERT-base | BoolQ + AltBoolQ | **76.12** | **74.97** | **75.57**
 | | | | 
BERT-large | BoolQ | 77.65 | 75.24 | 76.51 
BERT-large | AltBoolQ | 74.04 | 76.10 | 75.01 
BERT-large | BoolQ + AltBoolQ  | **77.86** | **77.17** | **77.53** 
 | | | | 
RoBERTa-base | BoolQ | **81.01** | 78.31 | **79.74** 
RoBERTa-base | AltBoolQ | 76.70 | 76.21 | 76.47 
RoBERTa-base | BoolQ + AltBoolQ | 79.91 | **78.48** | 79.24 
 | | | | 
RoBERTa-large | BoolQ | **86.18** | 82.85 | **84.61** 
RoBERTa-large | AltBoolQ | 84.65 | 82.92 | 83.84 
RoBERTa-large | BoolQ + AltBoolQ | 85.60 | **83.23** | 84.48 

Training on BoolQ generalizes to AltBoolQ and vice versa, suggesting little reliance on world knowledge. For BERT models, adding in AltBoolQ during training boosts performance.

## Example

Our method uses non-deep-learning approaches to randomize entities while attempting to preserve internal consistency. This can produce new examples that are factually false or weird, but that are still logically valid. For example:

```
Passage: Federal judiciary of the United States -- The federal courts are composed of three levels of courts. The Supreme Court of the United States is the court of last resort.
Question: is the federal court the same as the supreme court	
Answer: false
```

may become:
```
Passage: Global adminstration of the United States -- The global administrations are composed of three levels of administrations. The Vocal Administration of the United States is the administration of last resort.
Question: is the global administration the same as the vocal administration
Answer: false
```

As a note, redacting for MaskedBoolQ might make result in:

```
Passage: Redacted redacted of the United States -- The redacted redacted are composed of three levels of redacted. The Redacted Redacted of the United States is the redacted of last resort.
Question: is the redacted redacted the same as the redacted redacted
Answer: false
```

On its face, the passage and question are now unanswerable, but if one exploits statistical cues in the dataset, one could still guess correctly. This may suggest that the dataset is too easy.

## Method

An overview of the method:

1. **Grouping**. We use [fastText](https://fasttext.cc/) word embeddings to group entities with relationships that we cannot break, e.g. different forms of the same noun.
2. **Selection**. We use TF-IDF (Term Frequency-Inverse Document Frequency) for extracting relevant entities specific to each document.
3. **Randomization**. We use Gaussian Mixture Models to cluster the word embeddings. We use the clusters to resample words, and grouped words are shifted together, i.e. via word analogy, to preserve relationships within each group.

## Running experiments

For augmenting a dataset by either redacting or randomizing entities, see the [augment.ipynb](./augment.ipynb) jupyter notebook. For running and evaluating models on various datasets, see the [train.ipynb](./train.ipynb) jupyter notebook. Note: these are mostly for documentation and probably won't run correctly out of the box.

## License
All code is licensed under the MIT License. See the [LICENSE](./LICENSE) file.
