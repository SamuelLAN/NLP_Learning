#!/usr/bin/Python
# -*- coding: utf-8 -*-
import math
import numpy as np
from config.param import IS_TRAIN
from lib.nn_model_base import NN as NNBase


class NN(NNBase):

    def train_generator(self, train_object, val_object):
        """ Train model with all data loaded in memory """
        batch_size = self.params['batch_size']
        x_example, y_example = train_object.next_batch(batch_size)
        self.before_train(train_object.len, x_example, y_example)

        self.model.fit(x_example, y_example, epochs=0)

        if IS_TRAIN:
            # The returned value may be useful in the future
            history_object = self.model.fit_generator(train_object.batch_generator(batch_size),
                                                      epochs=self.params['epoch'],
                                                      steps_per_epoch=math.ceil(train_object.len * 1.0 / batch_size),
                                                      validation_data=val_object.batch_generator(batch_size),
                                                      validation_steps=math.ceil(val_object.len * 1.0 / batch_size),
                                                      callbacks=self.config_for_keras['callbacks'],
                                                      class_weight=self.__class_weight,
                                                      initial_epoch=self.__initial_epoch,
                                                      verbose=2)

            best_model_path = self.__get_best_model_path()
            if best_model_path:
                self.load_model(best_model_path)

        return self.test_generator(val_object)

    def test_generator(self, data_object, name='val'):
        """ evaluate the model performance while data size is big """
        # variables that record all results
        logits_list = []
        y_list = []

        # calculate the total steps
        batch_size = self.params['batch_size']
        steps = int(math.ceil(data_object.len * 1.0 / batch_size))

        # traverse all data
        for step in range(steps):
            tmp_x, tmp_y = data_object.next_batch(batch_size)
            # logits_list.append(self.predict(tmp_x))
            logits_list.append(tmp_y)
            y_list.append(tmp_y)

        logits_list = np.vstack(logits_list)[:data_object.len]
        y_list = np.vstack(y_list)[:data_object.len]

        return self.measure_and_print(y_list, logits_list, name)

    def evaluate_generator(self, data_object):
        batch_size = self.params['batch_size']
        steps = math.ceil(data_object.len * 1.0 / batch_size)
        return self.model.evaluate_generator(data_object.batch_generator(batch_size),
                                             steps=steps, workers=1, use_multiprocessing=False)

    def predict_generator(self, data_object):
        batch_size = self.params['batch_size']
        steps = math.ceil(data_object.len * 1.0 / batch_size)
        return self.model.predict_generator(data_object.batch_generator(batch_size),
                                            steps=steps, workers=1, use_multiprocessing=False)[:data_object.len]

    def predict_class_generator(self, data_object):
        output = self.predict_generator(data_object)
        return np.argmax(output, axis=-1)

    def predict_correct_generator(self, data_object):
        # variables that record all results
        logits_list = []
        y_list = []

        # calculate the total steps
        batch_size = self.params['batch_size']
        steps = int(math.ceil(data_object.len * 1.0 / batch_size))

        # traverse all data
        for step in range(steps):
            tmp_x, tmp_y = data_object.next_batch(batch_size)
            logits_list.append(self.predict(tmp_x))
            y_list.append(tmp_y)

        logits_list = np.argmax(np.vstack(logits_list), axis=-1)[:data_object.len]
        y_list = np.argmax(np.vstack(y_list), axis=-1)[:data_object.len]
        return logits_list == y_list
