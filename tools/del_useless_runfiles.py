#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys
from shutil import rmtree

# for running in the cmd, otherwise import error will occur
CUR_PATH = os.path.split(__file__)[0]
sys.path.append(os.path.split(CUR_PATH)[0])

from config.path import PATH_BOARD_DIR, PATH_MODEL_DIR

""" To clear the useless run_files """

upper_model_dir = os.path.split(PATH_MODEL_DIR)[0]
upper_board_dir = os.path.split(PATH_BOARD_DIR)[0]

print('****************************************')
print('checking %s ... \n' % upper_board_dir)

for model_name in os.listdir(upper_board_dir):
    model_dir = os.path.join(upper_model_dir, model_name)
    board_dir = os.path.join(upper_board_dir, model_name)

    print('-------------------------------')
    print('checking %s ... \n' % model_dir)

    # if this model dir is empty, remove it
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        if os.path.isdir(model_dir):
            os.rmdir(model_dir)
            print('Remove %s' % model_dir)

        rmtree(board_dir)
        print('Remove %s' % board_dir)
        continue

    for _time in os.listdir(board_dir):
        time_model_dir = os.path.join(model_dir, _time)
        time_board_dir = os.path.join(board_dir, _time)

        print('\tchecking %s ... ' % time_board_dir)

        # this model dir is empty, remove it
        if not os.path.isdir(time_model_dir) or not os.listdir(time_model_dir):
            if os.path.isdir(time_model_dir):
                os.rmdir(time_model_dir)
                print('\tRemove %s' % time_model_dir)

            rmtree(time_board_dir)
            print('\tRemove %s' % time_board_dir)
            continue

    # check again after remove some useless dir
    if not os.listdir(model_dir):
        os.rmdir(model_dir)
        os.rmdir(board_dir)

        print('Remove %s' % model_dir)
        print('Remove %s' % board_dir)

print('\nDone')
