import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model, Model
from pathlib import Path
from random import randint
import os
import sys
import time
from datetime import date, datetime
import numpy as np
import config as c
from lib.model import *
from lib.my_logging import *
from lib.trainer import Trainer
from lib.extFunc import loadArrFromFile, load_dictionary

CUDA_DEVICE_INDEX = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICE_INDEX
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

########################################
#            INITIALIZATION            #
########################################
use_relu = True
# use_relu = False

bool_load_model = True
# bool_load_model = False

CONST_SAVED_MODEL_OUTDIR = "mnist_tf_e2_202307171600_0"
CONST_LOAD_MODEL_DIR = "saved_model_label"
CONST_LOAD_PATTERN_MODEL_DIR = "saved_pattern_model_label"
CONST_SAVED_MODEL_DIR = f"./saved_models/{CONST_SAVED_MODEL_OUTDIR}"

filename = f"{CONST_SAVED_MODEL_DIR}/configuration.json"
config = load_dictionary(filename)
batch_size = config["batch_size"]
num_classes = config["num_classes"]

splits = CONST_SAVED_MODEL_OUTDIR.split("_tf")
dataset_name = splits[0]

if dataset_name not in ["mnist", "fashion_mnist", "kmnist"]:
    print(f"No dataset name found: {dataset_name}")
    exit()

########################################
#               LOGGING                #
########################################
logname = f"{CONST_SAVED_MODEL_DIR}/mnist_testlog.log"
logger = createLogger(logname)

########################################
#            PRE FUNCTION              #
########################################
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


########################################
#           LOAD FUNCTION              #
########################################
def load_pattern(filename):
    pattern = loadArrFromFile(filename)
    return pattern


def load_pattern_model():
    models = []
    if bool_load_model:
        for i in range(num_classes):
            dir = f"{CONST_SAVED_MODEL_DIR}/{CONST_LOAD_PATTERN_MODEL_DIR}_{i}"
            model = load_saved_model(dir)
            models.append(model)

    return models

def load_ban_model(patterns, pattern_models):
    models = []
    inputs = []
    outputs = []
    if bool_load_model:
        # for i in range(1):
        for i in range(num_classes):
            dir = f"{CONST_SAVED_MODEL_DIR}/{CONST_LOAD_MODEL_DIR}_{i}"
            model = load_saved_model(dir)
            model.summary()
            input = model.inputs
            inputs.append(input)
            # --- If we use double nodes mode, we do not include the tanh activation function during the test ---
            if config.get("use_double_nodes") is not None and config["use_double_nodes"]:
                # out = model.layers[-2].output
                out = model.layers[-1].output
            else:
                out = model.layers[-1].output
                # output_layer_weights = model.layers[-1].get_weights()[0]
                output_layer_weights = None

            pattern = patterns[i]
            # pattern = tf.cast(tf.convert_to_tensor([pattern]), tf.float32)

            k_patterns = []
            for p in pattern:
                tmp = tf.cast(tf.convert_to_tensor([p]), tf.float32)
                k_patterns.append(tmp)
            ori_inputs = tf.cast(tf.squeeze(input, 0), "float32")
            pattern_model = pattern_models[i]
            out_pattern = pattern_model(k_patterns)
            print(i, "out_pattern", out_pattern)
            output = BanHead(out_pattern, use_relu, config)(out, ori_inputs, output_layer_weights)

            outputs.append(output)

        output = concatenate(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=output)

def get_each_label_dataset(ds_test):
    # Filter the data based on each label
    test_label_datasets = []
    for i in range(config["num_classes"]):
        test_ds = ds_test.filter(lambda img, label: label == i)
        test_ds = test_ds.batch(config["batch_size"])
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        test_label_datasets.append(test_ds)

    return test_label_datasets

########################################
#           MAIN FUNCTION              #
########################################
def main():
    filename = f"{CONST_SAVED_MODEL_DIR}/pattern.txt"
    patterns = load_pattern(filename)
    print(f"pattern: {patterns}")
    pattern_models = load_pattern_model()
    model = load_ban_model(patterns, pattern_models)

    model.summary()

    (ds_train, ds_test), ds_info = tfds.load(
            dataset_name,
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )


    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    test_label_datasets = get_each_label_dataset(ds_test)

    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    acc_list = []
    for x, y in ds_test:
        inputs = [x for i in range(num_classes)]
        # print(np.shape(inputs))
        logits = model(inputs)
        equalvalent_check = tf.cast(tf.equal(y, tf.argmin(logits, 1)), 'float32')
        # equalvalent_check = tf.cast(tf.equal(y, tf.argmax(logits, 1)), 'float32')
        accuracy = tf.reduce_mean(equalvalent_check)
        acc_list.append(accuracy)
        # print("LOGIT", logits)
        # print("y, yhat", y, equalvalent_check)
        print(f"accuracy: {accuracy}")
        # exit()


    label_acc_list = {}
    for index, ds in enumerate(test_label_datasets):
        if label_acc_list.get(index) is None:
            label_acc_list[index] = []
        for x, y in ds:
            inputs = [x for i in range(num_classes)]
            logits = model(inputs)
            # print(logits)
            equalvalent_check = tf.cast(tf.equal(y, tf.argmin(logits, 1)), 'float32')
            # equalvalent_check = tf.cast(tf.equal(y, tf.argmax(logits, 1)), 'float32')
            accuracy = tf.reduce_mean(equalvalent_check)
            label_acc_list[index].append(accuracy)

            print(f"accuracy label {index}: {accuracy}")


    print("final accuracy", np.mean(acc_list))
    for i in range(config["num_classes"]):
        print(f"final accuracy for class {i}", np.mean(label_acc_list[i]))



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total time = {(end_time - start_time)}')
    print(f"logger file: {logname}")



