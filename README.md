# dependency_parsing_tf
Tensorflow implementation of "A Fast and Accurate Dependency Parser using Neural Networks"


# Tensorboard
tensorboard --logdir=path of model variables' folder

example: tensorboard --logdir=<base dir>/dependency_parsing_tf/data/params_2017-09-18



### update

1. update the code to run in python 3 with tensorflow 1.1
2. substitute cPickle for pickle in python3
3. the apply gradient issues (tf.gradients -> optimizer.compute_gradients)
4. ​