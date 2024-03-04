from tensorflow.python.keras.layers import Dense, LSTM, CuDNNLSTM, Reshape, BatchNormalization, Input, Conv2D, \
    MaxPool2D, Lambda, Bidirectional, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import char_len, train, valid
import numpy as np


class Model1:
    def __init__(self):
        self.best_model = 'best_model.hdf5'
        self.check_point = ModelCheckpoint(filepath=self.best_model, monitor='val_loss', save_best_only=True,
                                           mode='auto')
        self.crnn()
        labels = Input(name='the_labels', shape=[31], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss = Lambda(self.ctc_cost, output_shape=(1,), name='ctc')(
            [self.outputs, labels, input_length, label_length])

        # fit and compile model
        model = Model(inputs=[self.inputs, labels, input_length, label_length], outputs=loss)

        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

        # model.fit(x=[train['img'], train['txt'], train['input_length'], train['label_length']],
        #           y=np.zeros(len(train['img'])),
        #           batch_size=256, epochs=25,
        #           validation_data=([valid['img'], valid['txt'], valid['input_length'], valid['label_length']],
        #                            [np.zeros(len(valid['img']))]), callbacks=[self.check_point])

    def crnn(self):
        self.inputs = Input(shape=(32, 128, 1))

        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(self.inputs)

        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)

        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)

        batch_norm_5 = BatchNormalization()(conv_5)

        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

        squeezed = Lambda(lambda x: backend.squeeze(x, 1))(conv_7)

        blstm_1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(squeezed)
        drop1 = Dropout(.25)(blstm_1)
        blstm_2 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(blstm_1)
        drop2 = Dropout(.25)(blstm_2)

        self.outputs = Dense(char_len + 1, activation='softmax')(blstm_2)

        return Model(self.inputs, self.outputs)

    def ctc_cost(self, args):
        y_pred, labels, input_length, label_length = args
        return backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
