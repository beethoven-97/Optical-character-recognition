from translate import translator
from data_load import Data_load
from preprocess import Preprocess, train, valid, mem, chars
from model import Model1
import numpy as np
from tensorflow.python.keras import backend

# print(translator('en', 'es', 'i am donkey'))
d = Data_load()
Preprocess(d.images)
m = Model1()
m1 = m.crnn()

# test
m1.load_weights(m.best_model)
preds = m1.predict(np.array(valid['img'][:10]))
decoder = backend.get_value(
    backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

# result
for i, x in enumerate(decoder):
    print('org=', valid['org_txt'][i])
    print('pred=', end='')
    for j in x:
        if int(j) != -1:
            print(chars[int(j)], end='')
    print('\n')
