#!/usr/bin/env python

'''
character LSTM for name or not in the text (Binary Classification Problem)
'''

import codecs
import glob
import os
import pickle
import random
import re
import string

import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

random.seed(100)


class DataPreparation:

    def __init__(self):
        self.vocab = {}
        self.upper_cap = 20
        self.split_ratio = 0.75
        self.data_points = 0
        self.class_points = 0
        self.pad_position = 'pre'  # post - possible
        self.pad_value = 0
        self.names = [1]
        self.common = [0]
        self.metadata='models/metadata.pkl'

    def save(self, type):
        # store necessary values for later evaluation
        meta = {
            'word2idx': self.vocab,
            'upper_cap': self.upper_cap,
            'classes': self.class_points
        }
        with codecs.open(self.metadata, 'wb') as metafile:
            pickle.dump(meta, metafile)
        print('Saved Metadata to file')

    def pad_sequence(self, data, position='post'):
        if len(data) <= self.upper_cap:
            # pad to upper_cap length based on the position of padding
            places_to_fill = self.upper_cap - len(data)
            if position == 'post':
                data.extend([self.pad_value] * places_to_fill)
                return data
            else:
                zeroes = [self.pad_value] * places_to_fill
                zeroes.extend(data)
                return zeroes
        else:
            return data

    def prepare(self, data):
        X, Y = [], []
        for d in data:
            # encode words to numbers
            tmp = [self.vocab[ch] for ch in d[0]]
            # pad sequence for fixed length vector representation
            tmp = self.pad_sequence(tmp, self.pad_position)
            # adding 3rd dimension that stores the word information (representation)
            tmp = [self.one_hot_encode(unit) for unit in tmp]
            X.append(tmp)
            # encode labels
            if d[1] == 'names':
                Y.append(self.names)
            else:
                Y.append(self.common)
        return X, Y

    def train_test_split(self, data):
        # data split
        random.shuffle(data)
        cap = int(self.split_ratio * self.data_points)
        train = data[:cap]
        test = data[cap:]
        return train, test

    def preprocess(self, data):
        # basic preprocessing
        data = re.sub('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', ' ', data, flags=re.I)
        data = re.sub('([.,!?()\:\-@])', r' \1 ', data)
        data = re.sub("\d+", " NUM  ", data)
        data = data.lower()
        data = "".join([i for i in data if i not in string.punctuation])
        data = re.sub("\s{2,}", " ", data, flags=re.I)
        return data

    def remove_duplicates(self, data):
        # remove redundant
        dataset = [list(tupl) for tupl in {tuple(item) for item in data}]
        return dataset

    def set_vocabulary(self, data):
        # vocabulary builder
        words = [txt[0] for txt in data]
        chars = set("".join(words))
        # print(chars)
        self.vocab = dict((c, i + 1) for i, c in enumerate(chars))
        self.vocab.update({'OOV': 100})

    def one_hot_encode(self, data):
        # create array of vocabulary length
        hot_vec = np.zeros(len(self.vocab))
        # put 1 for the current char.
        hot_vec[data] = 1
        return hot_vec

    def get_count(self, data):
        return len(data)

    def load_data(self, data):
        dataset, classes = list(), list()
        # list of data files
        files = glob.glob(data + '/*.txt')
        for file in files:
            filename = file.split('/')[-1].split('.')[0]
            classes.append(filename)
            with codecs.open(file, 'r', 'utf-8') as infile:
                for line in infile:
                    try:
                        if len(line) < self.upper_cap:
                            # preprocess the text for normalized representation
                            line = self.preprocess(str(line).strip())
                            dataset.append([line, filename])
                    except:
                        pass

        # remove duplicates
        dataset = self.remove_duplicates(dataset)
        # shuffling the dataset to avaoid sequence baisness'''
        random.shuffle(dataset)

        self.data_points = self.get_count(dataset)
        self.class_points = self.get_count(classes)
        print(classes)
        print('Total data points: {}'.format(self.data_points))
        print('Total classes: {}'.format(self.class_points))
        return dataset


class CharLSTMModel(DataPreparation):

    def __init__(self):
        self.epochs = 10
        self.batch_size = [128, 256, 512, 1024]
        self.units = [64, 128, 256]
        self.dropout = 0.4
        self.activation = 'sigmoid'
        self.loss = 'binary_crossentropy'
        self.validation = 0.15
        self.verbose = 1
        self.embedding_dim = 300
        self.nn_model = "models/char-nn-ner.json"
        self.nn_model_weights = "models/char-nn-ner.h5"

    def compile(self, trX, trY, teX, teY, cap, vocab, bi=False, preembed=False):
        model = keras.models.Sequential()
        '''
        if preembed:
        	model.add(keras.layers.Embedding(vocab+1, 300, weights=[self.embedding_matrix],
        		                             input_length=20, trainable=False))
        else:
        	model.add(keras.layers.Embedding(vocab+1, 50, input_length=20))   # optional
        '''
        if bi:
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=(20, 28)))
        else:
            model.add(keras.layers.LSTM(128, input_shape=(20, 28), return_sequences=True))

        model.add(keras.layers.core.Dropout(0.30))

        if bi:
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False)))
        else:
            model.add(keras.layers.LSTM(128, return_sequences=False))

        model.add(keras.layers.core.Dropout(0.30))

        model.add(keras.layers.Dense(1, activation=self.activation))
        model.compile(loss=self.loss, optimizer='adam', metrics=['accuracy'])
        model.summary()
        model.fit(trX, trY, batch_size=256, epochs=self.epochs, \
                  verbose=self.verbose, validation_split=self.validation)

        score, acc = model.evaluate(teX, teY, verbose=self.verbose)
        print(('Test accuracy:', acc))
        self.dump_model_and_weights(model)

    def dump_model_and_weights(self, model):
        model_json = model.to_json()
        with open(self.nn_model, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(self.nn_model_weights)
        print("Saved model to disk")

    def load_model(self):
        json_file = open(self.nn_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.nn_model_weights)
        print("Loaded model from disk")
        return loaded_model


class FeatureExtractor:

    def __init__(self):
        self.feat = 1
        self.analyze_level = 'char_wb'
        self.lowercase = True
        self.vectorizer = CountVectorizer(
            stop_words='english',
            lowercase=self.lowercase,
            ngram_range=(1, self.feat),
            analyzer=self.analyze_level
        )

    def vectorize(self, data, dtype='train'):
        if dtype == 'train':
            vector = self.vectorizer.fit_transform(data)
        else:
            vector = self.vectorizer.transform(data)
        return vector

    def preprocess(self, data):
        data = re.sub('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', ' ', data, flags=re.I)
        data = re.sub('([.,!?()@\:\-])', r' \1 ', data)
        data = re.sub("\d+", " NUM ", data.lower())
        data = data.lower()
        data = "".join([i for i in data if i not in string.punctuation])
        data = re.sub("\s{2,}", " ", data, flags=re.I)
        return data

    def prepare(self, data, dtype='train'):
        text = [self.preprocess(i[0]) for i in data]
        if dtype == 'train':
            vec = self.vectorize(text)
        else:
            vec = self.vectorize(text, 'test')
        lab = []
        for d in data:
            label = d[1]
            if label == 'names':
                label = 1
            else:
                label = 0
            lab.append(label)
        return vec, lab


class SVMModel(FeatureExtractor):

    def __init__(self):
        self.svm_model = 'models/svm-model.pkl'

    def compile(self, trainX, trainY, testX, testY):
        model = SVC()
        model.fit(trainX, trainY)

        '''TRAINING DATA'''
        predictions = model.predict(trainX)
        a = np.array(trainY)
        count = 0
        for i in range(len(predictions)):
            if predictions[i] == a[i]:
                count = count + 1
        accuracy_train = count / float(len(predictions))

        '''TESTING DATA'''
        a = np.array(testY)
        predictions_test = model.predict(testX)
        count = 0
        for i in range(len(predictions_test)):
            if predictions_test[i] == a[i]:
                count = count + 1
        accuracy_test = count / float(len(predictions_test))

        print(('Training Accuracy: {}'.format(accuracy_train)))
        print(('Testing Accuracy: {}'.format(accuracy_test)))

        self.model_dump(model)

    def model_dump(self, model):
        joblib.dump(model, self.svm_model)
        print('Model Dumped')

    def model_load(self):
        model = joblib.load(self.svm_model)
        print('Model Loaded')
        return model

BASE_DIR = ''
DATA_DIR = os.path.join(BASE_DIR, 'data')

if __name__ == '__main__':
    # instances
    prepare_nn = DataPreparation()
    prepare_svm = FeatureExtractor()

    # load data
    dataset = prepare_nn.load_data(DATA_DIR)
    prepare_nn.set_vocabulary(dataset)
    print(('Vocabulary: {}'.format(len(prepare_nn.vocab))))
    # print(prepare_nn.vocab.keys())
    # train-test split
    train, test = prepare_nn.train_test_split(dataset)

    # NN preparation step
    trainX_nn, trainY_nn = prepare_nn.prepare(train)
    testX_nn, testY_nn = prepare_nn.prepare(test)
    prepare_nn.save(type='meta')

    # SVM preparation step
    trainX_svm, trainY_svm = prepare_svm.prepare(train)
    testX_svm, testY_svm = prepare_svm.prepare(test, 'test')
    pickle.dump(prepare_svm.vectorizer, open('models/svm-vocab.pkl', 'wb'))

    print(('Training data dimensions (Text): {}'.format(np.asarray(trainX_nn).shape)))
    print(('Training data dimensions (Labels): {}'.format(np.asarray(testY_nn).shape)))
    print(('Testing data dimensions (Text): {}'.format(np.asarray(testX_nn).shape)))
    print(('Testing data dimensions (Labels): {}'.format(np.asarray(testY_nn).shape)))

    ##############################################################################################
    # model-1

    print('Training LSTM - Model 1')
    ch_model = CharLSTMModel()

    ch_model.compile(np.array(trainX_nn), np.array(trainY_nn), np.array(testX_nn), \
                                  np.array(testY_nn), prepare_nn.upper_cap, len(prepare_nn.vocab), False, False)

    loaded_model_nn = ch_model.load_model()

    # model-2
    print('Training SVM - Model 2')
    ch_svm_model = SVMModel()
    ch_svm_model.compile(trainX_svm, trainY_svm, testX_svm, testY_svm)
    loaded_model_svm = ch_svm_model.model_load()
    print ("\n\n")
    