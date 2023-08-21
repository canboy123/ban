import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, MaxPooling2D, ReLU, Flatten, Dense, Add, \
    Concatenate, Embedding, LayerNormalization, MultiHeadAttention, Dropout
from pathlib import Path

########################################
#          SAVE/LOAD FUNCTION          #
########################################
def save_model(model, layer_index, CONST_SAVED_MODEL_DIR):
    # Save the trained model
    Path(f"{CONST_SAVED_MODEL_DIR}_{layer_index}").mkdir(parents=True, exist_ok=True)
    print("SAVE MODEL PATH", f"{CONST_SAVED_MODEL_DIR}_{layer_index}")
    try:
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}", save_format="tf")
    except Exception:
        print("exception saved model to h5 format")
        model.save(f"{CONST_SAVED_MODEL_DIR}_{layer_index}/saved_model.h5")

def load_saved_model(path):
    print("LOAD MODEL PATH", path)
    if path != "" and path is not None:
        model = load_model(f"{path}")

    return model


########################################
#            MODEL/LAYER               #
########################################
class BanHead(tf.keras.Model):
    def __init__(self, pattern, use_relu=False, config={}):
        super(BanHead, self).__init__()
        self._pattern = pattern
        self._use_relu = use_relu
        self._config = config

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        if self._use_relu:
            # --- Add the relu here
            inputs = tf.keras.layers.ReLU()(inputs)
            dist = tf.keras.layers.ReLU()(self._pattern)
        else:
            dist = self._pattern

        # KL-divergence loss
        if self._config["loss"] == "kl":
            kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
            pattern_probs = tf.nn.softmax(dist, axis=-1)
            logits_probs = tf.nn.softmax(inputs, axis=-1)
            x = kl(pattern_probs, logits_probs)
            # x = kl(dist, inputs)
            x = tf.expand_dims(x, axis=-1)
        # MSE loss
        elif self._config["loss"] == "mse":
            if self._config["num_of_pattern_per_label"] == 1:
                x = (inputs - dist) ** 2
                x = tf.reduce_mean(x, 1, keepdims=True)
            else:
                x = (inputs - dist) ** 2
                x = tf.reduce_mean(x, 0)
                x = tf.reduce_mean(x, 1, keepdims=True)

        return x

class PatternModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(PatternModel, self).__init__(**kwargs)
        self._config = config

    def build(self, input_shape):
        self.variable = tf.Variable(
            initial_value=tf.ones(shape=(self._config["output_nodes"],)),
            trainable=True if self._config["use_flexible_pattern"] is True else False
        )
        # self._dense = Dense(self._config["output_nodes"])
        super(PatternModel, self).build(input_shape)

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        return inputs * self.variable
        # return self._dense(inputs)

def createPatternModel(config):
    output_nodes = config["output_nodes"]

    inputs = []
    outputs = []
    for i in range(config["num_of_pattern_per_label"]):
        input = Input(shape=output_nodes)
        inputs.append(input)
        output = PatternModel(config)(input)
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def createNormalModel(config):
    input_shape = config["input_shape"]
    use_bias = config["use_bias"]
    activation = config["activation"]
    output_nodes = config["output_nodes"]

    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    out = Dense(output_nodes, use_bias=use_bias)(x)

    if activation == "sigmoid":
        out = tf.keras.activations.sigmoid(out)
    elif activation == "tanh":
        out = tf.keras.activations.tanh(out)

    model = Model(inputs=inputs, outputs=out)

    return model

def createModel(config):
    model = createNormalModel(config)

    return model