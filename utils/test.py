from general_utils import get_minibatches
import numpy as np
import time

a = np.reshape(np.arange(1000), (200, 5))
b = np.reshape(np.ones(shape=(1000,)), (200, 5))
print(a.shape, b.shape)
gen = get_minibatches([a, b], minibatch_size=5, is_multi_feature_input=False)
sample = next(gen)
print(sample)
# so, it returns a generator
# is_multi_feature_input is a strange paramater

from general_utils import get_pickle
a = get_pickle("../data/dump/pos2idx.pkl")
print(type(a))

from general_utils import get_vocab_dict
a = ['a', 'b', 'c', 'de']
b = get_vocab_dict(a)
print(b)

from general_utils import Progbar
p = Progbar(10000)
for i in range(10000):
    if i %1000 == 0:
        p.update(i)
        # time.sleep(1)
