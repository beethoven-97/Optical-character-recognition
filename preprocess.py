import data_load
import numpy as np
import cv2
from collections import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

chars, char_len = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', 79
train, valid, mem = defaultdict(list), defaultdict(list), defaultdict(int, {chars[i]: i for i in range(char_len)})
max_label_len = 31


class Preprocess:
    def __init__(self, images):
        for i, obj in enumerate(images):
            # gray
            img = cv2.cvtColor(cv2.imread(obj.path), cv2.COLOR_BGR2GRAY)

            # resize
            img = cv2.resize(img, (128, 32))
            img = np.expand_dims(img, axis=2)
            # print(img.shape)

            # normalize
            img = img / 255.

            # show
            # cv2.imshow(obj.text, img)
            # cv2.waitKey(0)

            # split
            self.split(i, obj.text, img)
        # convert to numpy
        self.to_numpy()

    def to_numpy(self):
        for i in train.keys():
            if i != 'txt':
                train[i] = np.array(train[i])
            else:
                train[i] = pad_sequences(train[i], maxlen=31, padding='post', value=char_len)
        for i in valid.keys():
            if i != 'txt':
                valid[i] = np.array(valid[i])
            else:
                valid[i] = pad_sequences(valid[i], maxlen=31, padding='post', value=char_len)

    def encode_texts(self, text):
        return list(mem[i] for i in text)

    def split(self, i, txt, img):
        if i % 10 == 0:
            valid['org_txt'].append(txt)
            valid['label_length'].append(len(txt))
            valid['input_length'].append(31)
            valid['img'].append(img)
            valid['txt'].append(self.encode_texts(txt))
        else:
            train['org_txt'].append(txt)
            train['label_length'].append(len(txt))
            train['input_length'].append(31)
            train['img'].append(img)
            train['txt'].append(self.encode_texts(txt))
