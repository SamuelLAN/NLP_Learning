#!/usr/bin/Python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

keras = tf.keras


class Model:
    def __init__(self, model_url, trainable=True, if_transform_weights=True):
        tags = {"train"} if trainable else None
        self.module = hub.Module(model_url, trainable=trainable, tags=tags)
        self.__variable_map = {}
        self.__variables = []

        # transform tensor variables to numpy arrays
        if if_transform_weights:
            self.__sess_run_variables()

    def __sess_run_variables(self):
        """ Sess run variables so that they could by numpy arrays (Takes almost 40 seconds) """
        print('Start initializing module variables ...')

        for variable in self.module.variables:
            keras.backend.get_session().run(variable.initializer)
            self.__variables.append(keras.backend.get_session().run(variable))

        for name, val in self.module.variable_map.items():
            keras.backend.get_session().run(val.initializer)
            self.__variable_map[name] = keras.backend.get_session().run(val)

        print('Finish initializing module variables')

    @property
    def variables(self):
        return self.__variables

    @property
    def variable_map(self):
        return self.__variable_map

    def get_input_info(self, signature=None):
        return self.module.get_input_info_dict(signature=signature)

    def get_output_info(self, signature=None):
        return self.module.get_output_info_dict(signature=signature)

    def get_variable_list_by_name_list(self, name_list):
        return [self.variable_map[name] for name in name_list]

    def get_variable_dict_by_name_list(self, name_list):
        result = {}
        for name in name_list:
            result[name] = self.variable_map[name]
        return result

    def get_model_specify_inputs(self, inputs=None, signature=None, as_dict=True):
        return self.module(inputs=inputs, signature=signature, as_dict=as_dict)

    def get_model(self, inputs=None, signature=None):
        input_key = list(self.get_input_info(signature).keys())[0]
        output_key = list(self.get_output_info(signature).keys())[0]
        return self.get_model_specify_inputs({input_key: inputs}, signature)[output_key]

    def gen_model(self, top_model=None, bottom_model=None, name=''):
        fn = self.get_model

        class TempModel(keras.Model):
            def __init__(self, _top_model, _bottom_model, _name=''):
                super(TempModel, self).__init__(name=_name)
                self.__top_model = _top_model
                self.__bottom_model = _bottom_model

            def call(self, _inputs, training=True):
                last_layer_outputs = self.__top_model(_inputs) if not isinstance(self.__top_model, type(None)) \
                    else _inputs
                features = fn(last_layer_outputs)
                return self.__bottom_model(features) if not isinstance(self.__bottom_model, type(None)) \
                    else features

        return TempModel(top_model, bottom_model, name)

    def test(self):
        print('Start loading data ...')
        (train_x, train_y), (val_x, val_y) = keras.datasets.cifar10.load_data()
        train_y = np.squeeze(train_y, -1)
        val_y = np.squeeze(val_y, -1)
        train_y_one_hot = np.eye(10)[train_y]
        val_y_one_hot = np.eye(10)[val_y]
        print('Finish loading')

        # ******************* gen model ***********************
        print('Start generating model ...')

        top_model = None
        bottom_model = keras.Sequential([
            keras.layers.Dense(10, activation='softmax'),
        ])

        model = self.gen_model(top_model, bottom_model, name='test_model')

        print('Finish generating model')

        # *********************** initialize **************************
        print('Start initializing ...')

        global_step = tf.train.get_or_create_global_step()
        epochs = 2
        batch_size = 20
        steps_per_epoch = int(len(train_x) // batch_size)
        steps = steps_per_epoch * epochs
        base_learning_rate = 0.0001
        decay_rate = 0.4
        learning_rate = tf.train.exponential_decay(base_learning_rate,
                                                   global_step,
                                                   steps_per_epoch,
                                                   decay_rate,
                                                   True)

        print('\tcompiling ...')
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                      loss=keras.losses.binary_crossentropy,
                      metrics=[
                          keras.metrics.binary_accuracy,
                          keras.metrics.binary_crossentropy,
                      ])

        print('\tinitializing global initializer ...')
        keras.backend.get_session().run(tf.global_variables_initializer())

        print('Finish initializing')

        # ********************* train ****************************
        print('Start fitting model ...')

        model.fit(train_x, train_y_one_hot,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(val_x, val_y_one_hot),
                  verbose=2)

        print('Finish fitting')


# o_model = Model("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3", True, False)
# o_model.test()
