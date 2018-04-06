from data_tools import DataTools

import numpy as np

import tensorflow as tf

import keras
from keras import backend as K
from keras.layers import Lambda
from keras.models import Model
from keras.models import load_model
from keras.layers import Input,concatenate,Dropout,Dense
from keras.layers import Embedding
from keras.layers import Convolution1D,GlobalMaxPooling1D
from keras.layers import Concatenate
from keras.layers import Reshape

from gensim import models as GenModels
import math

class KejModel(object):

    WORD_DIM = 250
    CLASS_DIM = 30
    NUM_CLASSES = 11
    MAX_SENTENCE_LEN = 100
    multiple = 1.0

    dt = None
    relationsMapping = {'other': 0, 'locaA': 1, 'locAa': 2, 'med-ill': 3, 'ill-med': 4,
                        "clsaA": 5, "clsAa": 6, "w-c": 7, "c-w": 8, "cs-ef": 9, "ef-cs": 10}
    idx2relation = {0: 'other', 1: 'locaA', 2: 'locAa', 3: 'med-ill', 4: 'ill-med',
                        5: "clsaA", 6: "clsAa", 7: "w-c", 8: "c-w", 9: "cs-ef", 10: "ef-cs"}
    def __init__(self):
        self.dt = DataTools()

    def load_kej_model(self,modelpath):
        return load_model(modelpath)

    def build_new_model(self,n_out,max_sentence_len,max_position,embedding,printflag = False):
        print("build new model")
        num_filter = 140
        filter_length = 3
        position_dims = 80
        print(max_sentence_len)

        words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
        # words = Embedding(70000, self.WORD_DIM)(words_input)
        words = embedding(words_input)
        # if printflag:
        #     print(words)

        sdp_input = Input(shape=(max_sentence_len,), dtype='float32', name='sdp_input')
        print(sdp_input)
        # K.variable(K.random_uniform(self.))
        init_att = math.sqrt(6.0 / (self.WORD_DIM + self.CLASS_DIM))
        U = K.variable(K.random_uniform([self.WORD_DIM,self.CLASS_DIM],minval=-init_att,
                                          maxval=init_att,
                                          dtype=tf.float32),name="U")
        classes_matrix = K.variable(
            K.random_uniform([self.CLASS_DIM, self.NUM_CLASSES], dtype=tf.float32), name="classmatrix")
        G = tf.matmul(tf.reshape(words, [-1, self.WORD_DIM]), U, name="G")
        print(G)

        G = tf.reshape(tf.matmul(G, classes_matrix), [-1, self.MAX_SENTENCE_LEN, self.NUM_CLASSES], name="G")
        print(G)

        init_m = math.sqrt(6.0 / (self.MAX_SENTENCE_LEN + self.NUM_CLASSES))
        M = tf.Variable(tf.random_uniform([self.MAX_SENTENCE_LEN, self.NUM_CLASSES],
                                          minval=-init_m,
                                          maxval=init_m,
                                          dtype=tf.float32),
                        name="M")
        alpha = tf.expand_dims(tf.matmul(sdp_input, M), axis=1, name="alpha")
        print(alpha)
        alpha = tf.matmul(alpha, tf.transpose(G, [0, 2, 1]), name="alpha")
        print(alpha)
        alpha = tf.nn.l2_normalize(tf.squeeze(alpha, axis=1), axis=-1, name="alpha")
        print(alpha)

        alpha_v = tf.add(sdp_input, tf.scalar_mul(self.multiple, alpha), name="alpha_v")
        alpha = tf.matrix_diag(alpha_v, name="alpha")
        print(alpha)
        weighted_data = tf.matmul(alpha, words, name="weighted_data")
        print(weighted_data)


        distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
        distance1 = Embedding(max_position, position_dims)(distance1_input)
        print(distance1)

        distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
        distance2 = Embedding(max_position, position_dims)(distance2_input)
        print(distance2)

        # my_concat = Lambda(lambda x: K.concatenate([x[0], x[1],x[2],x[3]], axis=-1))
        # output = my_concat([words,distance1,distance2,alpha])
        outputtemp = tf.concat([words,distance1,distance2],axis=2)
        # outputtemp = concatenate([words,distance1,distance2])
        print(outputtemp)

        output = Convolution1D(filters=num_filter,
                                kernel_size=filter_length,
                                padding='same',
                                activation='relu',
                                strides=1)(outputtemp)
        print(output)

        output = GlobalMaxPooling1D()(output)
        print(output)

        output = Dropout(0.25)(output)
        output = Dense(50, activation='relu')(output)
        output = Dropout(0.25)(output)
        output = Dense(n_out,activation='softmax')(output)

        model = Model(inputs = [sdp_input, words_input,distance1_input,distance2_input],outputs=[output])
        model.compile(loss = 'sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        return model


    def train_model(self, tosave , savepath, num_epoch = 20, batch_size = 64, load = True):
        ys, tokenMatrix, positionMatrix1, positionMatrix2,sdpMatrix = self.dt.get_data('pkl/train2.pkl.gz')
        n_out = max(ys) + 1
        max_sentence_len = tokenMatrix.shape[1]
        print(max_sentence_len)
        max_position = max(np.max(positionMatrix1), np.max(positionMatrix2)) + 1
        if load:
            model = load_model('model/kej_model.h5')
        else:
            genmodel = GenModels.Word2Vec.load("w2vmodel/word2vec2.model")
            embedding = genmodel.wv.get_keras_embedding(False)
            model = self.build_new_model(n_out,max_sentence_len,max_position,embedding)
        print("Start training")
        max_prec,max_rec,max_acc,max_f1 = 0,0,0,0
        # for epoch in range(num_epoch):
        model.fit([sdpMatrix, tokenMatrix,positionMatrix1,positionMatrix2],ys,batch_size=batch_size,verbose=True,epochs = num_epoch)
        if tosave:
            model.save(savepath)
            print("保存成功")

    def predict_classes(self,prediction):
        return prediction.argmax(axis=-1)

    def model_predict(self,testpkl_path,model_path = 'model/kej_model.h5'):
        ys, tokenMatrix, positionMatrix1, positionMatrix2 = self.dt.get_data(testpkl_path)
        model = load_model(model_path)
        pred_test = self.predict_classes(model.predict([tokenMatrix, positionMatrix1, positionMatrix2], verbose=False))
        acc = np.sum(pred_test == ys) / float(len(ys))
        print("测试样例数量: " + str(len(ys)))
        print("准确率: " + str(acc))
        print("错误样例：")
        wrong_outputs = []
        for i in range(len(ys)):
            if pred_test[i] != ys[i]:
                wrong_outputs.append(str(i) + " " + self.idx2relation[ys[i]] + " " + self.idx2relation[pred_test[i]])
        for item in wrong_outputs:
            print(item)

    def model_predict_one(self,model,relationidx, positionMatrix1, positionMatrix2, tokenMatrix):
        pred_test = self.predict_classes(model.predict([tokenMatrix, positionMatrix1, positionMatrix2], verbose=False))
        print(self.idx2relation[pred_test[0]])