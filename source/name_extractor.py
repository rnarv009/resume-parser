import pickle
import re
import keras
import numpy as np
import os
from flask import Flask, request
from sklearn.externals import joblib
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
from source.char_nn_ner_trainer import DataPreparation, FeatureExtractor  # , CharLSTMModel, SVMModel

class Predictor:

    def __init__(self):
        self.nn_model = "source/models/char-nn-ner.json"
        self.nn_model_weights = "source/models/char-nn-ner.h5"
        self.svm_model = 'source/models/svm-model.pkl'
        self.nn_model, self.svm_model = self.load(self.nn_model, self.nn_model_weights, self.svm_model)
        self.common = self._load_data('source/data/temp_new.txt')
        self.names = self._load_data('source/data/names_new.txt')
        self.prepare_nn = DataPreparation()
        self.prepare_svm = FeatureExtractor()
        self.svm_vocab = "source/models/svm-vocab.pkl"
        self.vectorizer = self.load_vocab(self.svm_vocab)
        self.nn_vocab = "source/models/metadata.pkl"
        self.vocab_nn = self.load_vocab(self.nn_vocab)

    @staticmethod
    def load_vocab(v):
        # logger.info('Load vocabulary')
        return pickle.load(open(v, "rb"))

    @staticmethod
    def _load_data(path):
        # logger.info("Loading from {}".format(path))
        data = []
        with open(path, 'r') as infile:
            for line in infile:
                data.append(line.strip().lower())
        return data

    def load(self, nn_model, nn_weights, svm_model):
        '''loader'''
        # loading NN model
        loaded_model_nn = self.load_model(nn_model, nn_weights)
        # loading SVM model
        loaded_model_svm = self.model_load()
        return loaded_model_nn, loaded_model_svm

    @staticmethod
    def load_model(definition, weight):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # print(dir_path)
        json_file = open(definition, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weight)
        # logger.info("Loaded Neural Network model from disk")
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return loaded_model

    @staticmethod
    def get_majority(nn, svm):
        '''count scheme'''
        if svm == nn:
            return svm
        else:
            return nn

    def model_load(self):
        model = joblib.load(self.svm_model)
        # logger.info('Loaded SVM model from disk')
        return model

    def vectorize(self, data):
        return self.vectorizer.transform(data)

    def one_hot_encode(self, data):
        '''encode to one-hot (3-D in the data)'''
        # create array of vocabulary length
        hot_vec = np.zeros(len(self.vocab_nn['word2idx']))
        # put 1 for the current char.
        hot_vec[data] = 1
        return hot_vec

    def in_gazette(self, w, typeOf):
        if typeOf == 'common':
            if w in self.common:
                return 1
            else:
                return 0
        elif typeOf == 'name':
            if w in self.names:
                return 1
            else:
                return 0

    @staticmethod
    def format_output(sample, extraction):
        '''Output formatter'''
        # logger.info("Formatting Output")
        # print("extraction:",extraction)
        final = []
        for extract in range(len(extraction)):
            if extraction[extract] == 'NOUN':
                final.append('<NOUN>')
                final.append(sample[extract])
                final.append('</NOUN>')
            else:
                final.append(sample[extract])
        final = re.sub('</NOUN>\s<NOUN>', '', ' '.join(final))
        final = re.sub('\s{2,}', ' ', final)
        # logger.info("Trying to match Initials Greedly")
        candidates = re.findall("(\\b[a-zA-Z]\\b)?\s*<NOUN>\s*(.+?)<\/NOUN>\s*(\\b[a-zA-Z]\\b)?", final)
        final = []

        for i in candidates:
            if i[2] not in ['f', 'm']:
                final.append('<NOUN>' + ' '.join(i).title() + '</NOUN>')
            else:
                final.append('<NOUN>' + ' '.join(i[:-1]).title() + '</NOUN>')

        # final = ['<NOUN>'+' '.join(i).title()+'</NOUN>' for i in candidates]
        final = ' '.join(final)
        # final = re.sub('</NOUN>\s<NOUN>','', ' '.join(final))
        final = re.sub('\s{2,}', ' ', final).strip()
        return final

    def test(self, samples):
        '''main test function'''
        extraction = []
        for sample in samples:
            # print(sample)
            sample = self.prepare_nn.preprocess(sample).split()
            # print(sample)
            for s in sample:

                # SVM test
                # logger.info("Predicting using SVM")
                text = self.prepare_svm.preprocess(s)
                vec = self.vectorize([text])
                pred = self.svm_model.predict(vec)[0]
                if pred == 1:
                    val_svm = 'NOUN'
                else:
                    val_svm = 'O'

                # NN test
                # logger.info("Predicting using NN")
                line = self.prepare_nn.preprocess(str(s).strip())
                # logger.info("Cleaned Sentence: {}".format(line))
                tmp = []
                for ch in line:
                    if ch in self.vocab_nn['word2idx']:
                        tmp.append(self.vocab_nn['word2idx'][ch])
                    else:
                        tmp.append(self.vocab_nn['word2idx']['OOV'])
                # logger.info("Word2Index: {}".format(tmp))
                tmp = self.prepare_nn.pad_sequence(tmp, 'pre')
                tmp = np.array([self.one_hot_encode(unit) for unit in tmp])
                tmp = np.reshape(tmp, (1, self.vocab_nn['upper_cap'], len(self.vocab_nn['word2idx'])))
                pred = self.nn_model.predict(tmp)[0]
                '''threshold probability (showing model confidence)'''
                if pred >= 0.59:
                    # logger.info("Looking in Common Word Gazetteer")
                    if self.in_gazette(s, 'common'):
                        val_nn = 'O'
                    else:
                        val_nn = 'NOUN'
                else:
                    # logger.info("Looking in Names Gazetteer")
                    if self.in_gazette(s, 'name'):
                        val_nn = 'NOUN'
                    else:
                        val_nn = 'O'
                # print((s, pred))
                val = self.get_majority(val_nn, val_svm)  # raw vote # upgrade to prob.
                extraction.append(val)
        return self.format_output(sample, extraction)  # format output of the API




if __name__ == '__main__':
    test_app = Predictor()
    output = test_app.test(['rahul verma is from varanasi'])
    print(output)
