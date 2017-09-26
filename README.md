# dependency_parsing_tf
Tensorflow implementation of "A Fast and Accurate Dependency Parser using Neural Networks"
https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf

# Tensorboard
tensorboard --logdir=path of model variables' folder

example: tensorboard --logdir=<base dir>/dependency_parsing_tf/data/params_2017-09-18

# Recent changes
1. transition to ****tf 1.2****
2. added cube activation function (ref: paper)
3. trainable word embeddings - initialized with 50d word2vec
4. l2 loss for regularization (ref: paper)
5. tensorboard visualization
6. ****Dev UAS****: 90.03 ****Test UAS****: 90.42
7. No functionality for LAS currently. it can be done with few changes in feature_extraction.py. I will try to add it.

# training (exisiting dataset)
python parser_model.py

# For new dataset
1. Build new vocabulary & embedding matrices -> set "load_existing_dump=False" in parser_model.py. This will overwrite existing "data/dump" directory content
2. python parser_model.py

# training dataset
CONLL format


