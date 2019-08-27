#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import numpy as np
import lightgbm as lgb
from config.path import PATH_MODEL_DIR, mkdir_time


class LGB:
    '''
    LightGBM
    '''

    params = {
        'max_depth': -1,
        'n_estimators': 999999,
        'learning_rate': 0.02,
        'colsample_bytree': 0.3,
        'num_leaves': 5,
        'boosting_type': 'goss',  # gbdt or goss,
        'metric': 'binary_error,auc,binary_logloss,l2',
        'objective': 'binary',  # regression_l2 or binary or xentropy
        # 'reg_lambda': 0.1,
        'n_jobs': -1
    }

    def __init__(self, model_dir, model_name=None):
        # get the time from model name and create a directory for this time
        dir_path = mkdir_time(PATH_MODEL_DIR, model_dir)
        self.__model_path = os.path.join(dir_path, model_name + '.model' if model_name else 'lgb.model')
        if model_name and os.path.isfile(self.__model_path):
            self.__has_train = True
            self.__model = lgb.Booster(model_file=self.__model_path)
        else:
            self.__has_train = False
            self.__model = lgb.LGBMClassifier(**self.params)

    def train(self, train_x, train_y, val_x, val_y):
        if self.__has_train:
            return
        print('train ...')
        self.__model.fit(train_x, train_y,
                         eval_set=[(val_x, val_y)],
                         verbose=True,
                         early_stopping_rounds=2000)

        self.save()

    def predict_class(self, X):
        if self.__has_train:
            return self.__model.predict(X) >= 0.5
        return np.argmax(self.__model.predict_proba(X), axis=1)

    def predict_prob(self, X, class_no=1):
        if self.__has_train:
            return self.__model.predict(X)[:, class_no]
        return self.__model.predict_proba(X)[:, class_no]

    def save(self):
        if self.__has_train:
            return
        self.__model.booster_.save_model(self.__model_path)
        print('Finish saving model to %s' % self.__model_path)
