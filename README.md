# Pretraining of Transformers on Question Answering without External Data

## Overview

This repository contains the code for the [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) final project in Winter 2021 titled "**Pretraining of Transformers on Question Answering without External Data**." The final report can be found [here](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/reports/final_reports/report253.pdf).

The project was submitted as a default final project (IID SQuAD track), which had some interesting [rules](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/project/default-final-project-handout-squad-track.pdf):
1. Build a QA system for SQuAD 2.0.
2. **No pre-existing implementations allowed unless written yourself, i.e. everything must be written from scratch.**
3. No pretrained contextual embeddings allowed (e.g. BERT, ELMO, GPT, etc.). GloVe was allowed.
4. The official dev set is split into a new dev set and test set.
5. The test set is hidden and used for leaderboards. No peeking at the official dev set!
6. **No additional data sources whatsoever.** Only the training set, provided dev set, GloVe, and provided starter code allowed.

This project ambitiously tried to see how well recent Transformer-based pretraining approaches fare in this ultra low-resource setting. It contains RoBERTa/ELECTRA style approaches written from scratch. Most other submissions likely used architectures or variations of architectures with stronger inductive biases like BiDAF, QANet, and so on.

Spoiler: BERT/RoBERTa/ELECTRA-style approaches have difficulty predicting more than the majority class. Even after tuning down the vocabulary size from 50k to 5k and using data augmentation, the model is not able to beat the provided BiDAF baseline. Better data-efficient pretraining approaches are needed!

## Findings
There were several takeaways from this project:
1. Pre-norm RoBERTa is easier to train than post-norm as number of layers increase.
2. ELECTRA-style pretraining is indeed faster than RoBERTa.
3. Smaller vocab sizes are required in this setting (5k vs the normal 50k), because otherwise models don't see enough samples of each token during training.
4. Models struggled more with deciding between an answer and N/A, than finding which span of text to answer with. Data augmentation to balance classes thus helped quite a bit.

Results summary:

Model | Dev F1 | Dev EM | Test F1 | Test EM
--- | :---: | :---: | :---: | :---: 
Majority class | 52.17 | 52.17 | - | -
BiDAF (baseline) | 60.79 | 57.50 |  60.32 | 56.81
RoBERTa post-norm (50k vocab) | Failed | Failed | - | -
\+ pre-norm | 51.37 | - | - | -
\+ ELECTRA-style | 51.82 | - | - | -
\+ 5k vocab  | 56.86 | 53.64 | - | -
\+ Data augmentation | 60.46 | 57.86 | 56.96 | 54.56

Even with all of these modifications, the RoBERTa/ELECTRA approaches did not beat out the baseline.

## Methods
This project contains from-scratch PyTorch implementations of:

1. A RoBERTa-style transformer model with options for post-norm or pre-norm settings, [here](./models/roberta.py) and [here](./models/transformer.py).
2. Byte-level byte pair encoding (BPE), [here](./preprocess/bpe.py).
3. RoBERTa and ELECTRA-style pretraining [here](./trainer/roberta_pretrainer.py) and [here](./trainer/electra_pretrainer.py).

It also contains code for some exploratory approaches like:
1. Adding more stages to the ELECTRA pretraining method (didn't work very well).
2. Data augmentation by adding new questions from the pretrained model (boosted F1 by about 4 points).

## How to run

Mostly for documentation, may not run correctly out of the box.

Running setup:

```
python setup.py bpe --data_sub_dir=bpe5k
```

Running training:

```
python train.py roberta_pretrain -n roberta_pretrain --batch_size=16 --gradient_accumulation=1 --n_layers=12 --num_epochs=1000 --lr=0.04 --warmup_steps=10000 --power_decay=-0.5 --decay_forever=True --prenorm=true --data_sub_dir=bpe5k --mask_prob=0.25
```

## License

All code is released under the MIT license. See the [LICENSE](./LICENSE) file.