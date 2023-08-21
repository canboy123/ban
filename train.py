import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model, Model
from pathlib import Path
from random import randint
import os
import wandb
import sys
import time
from datetime import date, datetime
import numpy as np
import config as c
from lib.my_logging import *
from lib.model import *
from lib.trainer import Trainer
from lib.extFunc import saveArrToFile, loadArrFromFile, save_dictionary, load_dictionary

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
config = c.config
bool_save_model = True
# bool_save_model = False

bool_load_model = True
bool_load_model = False

# If load_model_for_retraining is True, then we will retrain all models
load_model_for_retraining = True
# load_model_for_retraining = False

inputs = tf.keras.Input(shape=config["input_shape"])
# dd/mm/YY
if bool_load_model is not True:
    d1 = datetime.now().strftime("%Y%m%d%H00_"+CUDA_DEVICE_INDEX)
else:
    d1 = "202306161500_2"

start_index = 9

CONST_SAVED_MODEL_OUTDIR = f"""./saved_models/{config["dataset_name"]}_tf_e{config["epochs"]}_{d1}"""
CONST_SAVED_MODEL_DIR = CONST_SAVED_MODEL_OUTDIR+"/"+"saved_model_label"
CONST_SAVED_PATTERN_MODEL_DIR = CONST_SAVED_MODEL_OUTDIR+"/"+"saved_pattern_model_label"
CONST_SAVED_FIG_DIR = CONST_SAVED_MODEL_OUTDIR+"/"+"imgs"
Path(f"{CONST_SAVED_MODEL_OUTDIR}").mkdir(parents=True, exist_ok=True)
Path(f"{CONST_SAVED_FIG_DIR}").mkdir(parents=True, exist_ok=True)

config["save_fig_path"] = CONST_SAVED_FIG_DIR
modelSummaryStr = ""
patternModelSummaryStr = ""

########################################
#                WANDB                 #
########################################
def initialize_wandb(config, model_index):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"""ban_{config["dataset_name"]}_tf_e{config["epochs"]}_{d1}""",
        name=f"M{model_index}",
        # track hyperparameters and run metadata
        config=config
    )

if bool_load_model:
    filename = f"{CONST_SAVED_MODEL_OUTDIR}/configuration.json"
    config = load_dictionary(filename)
    # config["learning_rate"] = 0.002
########################################
#               LOGGING                #
########################################
logname = f"""{CONST_SAVED_MODEL_OUTDIR}/{config["dataset_name"]}_log_{d1}.log"""
logger = createLogger(logname)

########################################
#           LOAD FUNCTION              #
########################################

def load_pattern(filename):
    pattern = loadArrFromFile(filename)
    return pattern

def load_pattern_model(model_index):
    models = []
    if bool_load_model:
        # for i in range(config["num_classes"]):
        #     if train_one_model is True and i != model_index:
        #         continue
        dir = f"{CONST_SAVED_PATTERN_MODEL_DIR}_{model_index}"
        model = load_saved_model(dir)
        # models.append(model)

    return model
    # return models

def load_models(dir):
    model = load_saved_model(dir)
    model.summary()
    output = model.layers[-1].output
    tmp_model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

    return tmp_model
########################################
#            OTHER FUNCTION            #
########################################
def generateClassPattern(num_of_nodes, index=0):
    patterns = []
    if config["use_same_pattern"]:
        tf.random.set_seed(5)

    for i in range(config["num_of_pattern_per_label"]):
        pattern = tf.random.uniform(shape=[num_of_nodes], minval=config["min_node"], maxval=config["max_node"])
        # pattern = tf.random.uniform(shape=[num_of_nodes], minval=index*0.001, maxval=index*0.001+0.0001)
        # pattern = tf.random.uniform(shape=[num_of_nodes], minval=index*10, maxval=index*10+1)
        # pattern = tf.random.uniform(shape=[num_of_nodes], minval=1, maxval=50)
        # threshold = (index*0.001) + (((index*0.001+0.0001) - (index*0.001)) / 2)
        # pattern = tf.where(rand_tensor > threshold, x=tf.ones_like(rand_tensor), y=tf.zeros_like(rand_tensor))

        patterns.append(pattern)

    return patterns

def getModelSummaryFromKeras(s):
    global modelSummaryStr
    modelSummaryStr += f"{s}\n"

def getPatternModelSummaryFromKeras(s):
    global patternModelSummaryStr
    patternModelSummaryStr += f"{s}\n"

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def getOptimizer():
    decay_steps = int(config["epochs"]*config["num_training_data"]/config["batch_size"])
    use_learning_rate = tf.keras.experimental.CosineDecay(config["learning_rate"], decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(use_learning_rate, momentum=0.9)

    return optimizer

def get_each_label_dataset(ds_test):
    # Filter the data based on each label
    test_label_datasets = []
    for i in range(config["num_classes"]):
        test_ds = ds_test.filter(lambda img, label: label == i)
        # test_ds = test_ds.batch(config["batch_size"])
        test_ds = test_ds.batch(256)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        test_label_datasets.append(test_ds)

    return test_label_datasets

def main():
    if bool_load_model:
        # --- Load the pattern and the pattern model ---
        filename = f"{CONST_SAVED_MODEL_OUTDIR}/pattern.txt"
        patterns = load_pattern(filename)
    else:
        patterns = []
        if config["use_same_pattern"]:
            pattern = generateClassPattern(config["output_nodes"])
            for i in range(config["num_classes"]):
                patterns.append(pattern)
        else:
            for i in range(config["num_classes"]):
                patterns.append(generateClassPattern(config["output_nodes"], i))

        np_patterns = []
        for label_patterns in patterns:
            tmpPatterns = []
            for pattern in label_patterns:
                tmp = []
                for i in pattern.numpy():
                    tmp.append(i)
                tmpPatterns.append(tmp)
            np_patterns.append(tmpPatterns)

        # --- Save the pattern into the directory ---
        filename = f"{CONST_SAVED_MODEL_OUTDIR}/pattern.txt"
        saveArrToFile(filename, np_patterns)
        logger.info(f'Pattern: {np_patterns}, {np.shape(np_patterns)}')

    # --- Save configuration ---
    filename = f"{CONST_SAVED_MODEL_OUTDIR}/configuration.json"
    save_dictionary(config, filename)

    (ds_train, ds_test), ds_info = tfds.load(
            config["dataset_name"],
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # ds_train = ds_train.batch(128)
    # ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Filter the data based on each label
    train_label_datasets = []
    for i in range(config["num_classes"]):
        train_ds = ds_train.filter(lambda img, label: label == i)
        train_ds = train_ds.batch(config["batch_size"])
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        train_label_datasets.append(train_ds)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.batch(128)
    # ds_test = ds_test.cache()
    # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    test_label_datasets = get_each_label_dataset(ds_test)

    print("START TRAINING ...")

    logits_per_class = []
    for index, ds in enumerate(train_label_datasets):
        if bool_load_model:
            # This is used to retrain only specific model index
            # If load_model_for_retraining is True, then we will retrain all models
            if index < start_index and load_model_for_retraining is False:
                continue
            dir = f"{CONST_SAVED_MODEL_DIR}_{index}"
            model = load_saved_model(dir)
            pattern_model = load_pattern_model(index)
        else:
            model = createModel(config)
            pattern_model = createPatternModel(config)

        pattern = patterns[index]
        optimizer = getOptimizer()
        pattern_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Print one of the model
        if index == 0:
            model.summary(print_fn=getModelSummaryFromKeras)
            logger.info(msg=f"1 class model summary:\n {modelSummaryStr}")

            pattern_model.summary(print_fn=getPatternModelSummaryFromKeras)
            logger.info(msg=f"pattern model summary:\n {patternModelSummaryStr}")

        initialize_wandb(config, index)
        print(f"Train on class: {index}")

        print(f"pattern: {pattern}")
        trainer = Trainer(model, optimizer, pattern, pattern_model, pattern_optimizer, model_index=index, config=config)
        trainer.start_training(ds, config["epochs"], wandb)

        if bool_save_model:
            # --- Save normal model ---
            save_model(model, index, CONST_SAVED_MODEL_DIR)
            # --- Save pattern model ---
            save_model(pattern_model, index, CONST_SAVED_PATTERN_MODEL_DIR)

        wandb.finish()



if __name__ == "__main__":
    logger.info(f'Configurations: {config}')
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total time = {(end_time - start_time)}')
    print(f"logger file: {logname}")
    print(f"saved model directory: {CONST_SAVED_MODEL_OUTDIR}")