import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, \
AveragePooling2D, Add, ReLU, PReLU, LeakyReLU, DepthwiseConv2D, ELU, GlobalAveragePooling2D, Concatenate

CONST_REGULARIZER = tf.keras.regularizers.l2(1e-4)
CONST_AVG_POOLING_POOL_SIZE = 2

class Trainer(object):
    """Trains a ResNetCifar10 model."""

    def __init__(self, models, optimizer, pattern, pattern_model, pattern_optimizer, model_index, config={}):
        """Constructor.

        Args:
          model: an instance of ResNetCifar10Model instance.
        """
        self._model = models
        self._model_index = model_index
        self._optimizer = optimizer
        self._pattern = pattern
        self._pattern_model = pattern_model
        self._pattern_optimizer = pattern_optimizer
        self._config = config

        self.weight_decay = 5e-4

    def start_training(self, train_dataset, epochs, wandb=None):
        @tf.function()
        def train_step(images, true_pattern):
            model = self._model
            optimizer = self._optimizer
            pattern_optimizer = self._pattern_optimizer
            pattern_model = self._pattern_model
            with tf.GradientTape(persistent=True) as tape:
                logits = model(images)
                pattern = pattern_model(true_pattern)

                mse = tf.keras.losses.MeanSquaredError()
                loss = 0
                # KL-divergence loss
                if self._config["loss"] =="kl":
                    kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
                    pattern_probs = tf.nn.softmax(pattern, axis=-1)
                    logits_probs = tf.nn.softmax(logits, axis=-1)
                    loss = kl(pattern_probs, logits_probs)
                # MSE loss
                elif self._config["loss"] =="mse":
                    loss = mse(pattern, logits)


                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                # gradients = [(tf.clip_by_norm(grad, 5.0)) for grad in gradients]

                pattern_losses = 0.0
                # --- We do not update the pattern model if we use double node mode ---
                if self._config["use_pattern"] is True and self._config["use_flexible_pattern"]:
                    #--- Pattern flexible variables loss ---
                    avg_pattern = tf.reduce_mean(logits, axis=0, keepdims=True)
                    length = tf.shape(images)[0]  # or tf.size(y_batch_train)
                    avg_pattern = tf.tile(avg_pattern, multiples=[length, 1])
                    # KL-divergence loss
                    if self._config["loss"] =="kl":
                        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
                        pattern_probs = tf.nn.softmax(pattern, axis=-1)
                        avg_pattern_probs = tf.nn.softmax(avg_pattern, axis=-1)
                        pattern_losses = kl(pattern_probs, avg_pattern_probs)
                    # MSE loss
                    elif self._config["loss"] =="mse":
                        pattern_losses = mse(pattern, avg_pattern)
                    # pattern_losses = tf.reduce_mean(mse(pattern, logits))
                    pattern_gradients = tape.gradient(pattern_losses, pattern_model.trainable_variables)
                    pattern_optimizer.apply_gradients(zip(pattern_gradients, pattern_model.trainable_variables))

            return loss, pattern_losses

        for epoch in range(epochs):
            start_time = time.time()
            total_model_loss = []
            total_pattern_model_loss = []

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # Duplicate the y_batch_train based on the number of models that we have
                length = tf.shape(y_batch_train)[0]  # or tf.size(y_batch_train)

                # Split the variable into 8 tensors along the first dimension
                split_variables = tf.split(self._pattern, num_or_size_splits=self._config["num_of_pattern_per_label"], axis=0)

                # Reshape each split variable to have size [1, 128]
                reshaped_variables = [tf.reshape(split_var, [1, -1]) for split_var in split_variables]

                # Copy each split variable to have size [128, 128]
                patterns = [tf.tile(reshaped_var, [length, 1]) for reshaped_var in reshaped_variables]

                model_loss, pattern_model_loss = train_step(x_batch_train, patterns)
                total_model_loss.append(np.mean(model_loss))
                total_pattern_model_loss.append(np.mean(pattern_model_loss))

            end_time = time.time()
            total_time_used = end_time - start_time
            msg = {
                "Epoch": f"{epoch+1}/{epochs}",
                "Model Index": self._model_index,
                "Model Loss": np.mean(total_model_loss),
                "Pattern Model Loss": np.mean(total_pattern_model_loss),
                "Time": f"{total_time_used}s"
            }

            # --- Log into wandb ---
            if wandb is not None:
                wandb_log = {
                    "Model Loss": np.mean(total_model_loss),
                    "Pattern Model Loss": np.mean(total_pattern_model_loss),
                    "Time": total_time_used
                }
                wandb.log(wandb_log)

            items = ', '.join(f'{key}: {value}' for key, value in msg.items())
            print(items)
