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
        self._loss = config["loss"]
        self._config = config

    def call(self, inputs, ori_inputs=None, output_layer_weights=None):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        ori_inputs = Flatten()(ori_inputs)

        # x = inputs - self._pattern
        # x = tf.reduce_sum(x, 1, keepdims=True)
        if self._use_relu:
            # --- Add the relu here
            inputs = tf.keras.layers.ReLU()(inputs)
            dist = tf.keras.layers.ReLU()(self._pattern)
        else:
            dist = self._pattern

        # KL-divergence loss
        if self._loss == "kl":
            kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
            pattern_probs = tf.nn.softmax(dist, axis=-1)
            logits_probs = tf.nn.softmax(inputs, axis=-1)
            x = kl(pattern_probs, logits_probs)
            # x = kl(dist, inputs)
            x = tf.expand_dims(x, axis=-1)
        # MSE loss
        elif self._loss == "mse":
            x = ((dist - inputs) ** 2)
            x = tf.reduce_mean(x, 1, keepdims=True)

        return x

class BanInputLayer(tf.keras.Model):
    def __init__(self, input_nodes, **kwargs):
        super(BanInputLayer, self).__init__(**kwargs)
        self._input_nodes = input_nodes

    def build(self, input_shape):
        # self.variable = tf.Variable(
        #     initial_value=tf.ones(shape=(self._input_nodes,)),
        #     trainable=True
        # )
        self.variable = tf.Variable(
            initial_value=tf.ones(shape=(input_shape[-1],)),
            trainable=True
        )
        super(BanInputLayer, self).build(input_shape)

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        x = inputs * self.variable

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

def createNormalModel(model_index, config):
    input_shape = config["input_shape"]
    use_bias = config["use_bias"]
    activation = config["activation"]
    output_nodes = config["output_nodes"]

    inputs = Input(shape=input_shape)
    # x = Conv2D(8, 3, 2)(inputs)
    # x = BatchNormalization()(x)
    # x = Conv2D(16, 3, 2)(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, 3, 2)(x)
    # x = BatchNormalization()(x)
    # x = Flatten()(x)
    # x = BanInputLayer(input_nodes)(x)
    x = Flatten()(inputs)
    out = Dense(output_nodes, use_bias=use_bias)(x)

    if activation == "sigmoid":
        out = tf.keras.activations.sigmoid(out)
    elif activation == "tanh":
        out = tf.keras.activations.tanh(out)

    model = Model(inputs=inputs, outputs=out)

    return model

def createModel(model_index, config):
    model = createNormalModel(model_index, config)

    return model

