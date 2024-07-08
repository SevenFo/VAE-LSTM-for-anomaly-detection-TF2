import tensorflow as tf
from tensorflow import keras
import tensorflow_probability.python.distributions as tfd
import numpy as np
from matplotlib import pyplot as plt

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


class Encoder(keras.Layer):
    def __init__(self, configs, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.configs = configs
        self.init = keras.initializers.GlorotUniform()
        self.conv_1 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 16,
            kernel_size=(3, self.configs["n_channel"]),
            strides=(2, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.conv_2 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 8,
            kernel_size=(3, self.configs["n_channel"]),
            strides=(2, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.conv_3 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 4,
            kernel_size=(3, self.configs["n_channel"]),
            strides=(2, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.conv_4 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"],
            kernel_size=(4, self.configs["n_channel"]),
            strides=1,
            padding="valid",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(
            units=self.configs["code_size"] * 4,
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.code_mean = keras.layers.Dense(
            units=self.configs["code_size"],
            activation=None,
            kernel_initializer=self.init,
            name="code_mean",
        )
        self.code_std_dev = keras.layers.Dense(
            units=self.configs["code_size"],
            activation=lambda x: tf.nn.relu(x) + 1e-2,
            kernel_initializer=self.init,
            name="code_std_dev",
        )

    def call(self, inputs, training=None, mask=None):
        x = tf.expand_dims(inputs, axis=-1)
        x = tf.pad(x, [[0, 0], [4, 4], [0, 0], [0, 0]], "SYMMETRIC")
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        code_mean = self.code_mean(x)
        code_std_dev = self.code_std_dev(x)
        mvn = tfd.MultivariateNormalDiag(loc=code_mean, scale_diag=code_std_dev)
        code_sample = mvn.sample()
        return code_sample, code_mean, code_std_dev

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({"configs": self.configs})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.Layer):
    def __init__(self, configs, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.configs = configs
        self.init = tf.keras.initializers.GlorotUniform()

        self.decoded_1 = keras.layers.Dense(
            units=self.configs["num_hidden_units"],
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )

        self.decoded_2 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"],
            kernel_size=1,
            padding="same",
            activation=tf.nn.leaky_relu,
        )
        self.decoded_3 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 4,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.decoded_4 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 8,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )
        self.decoded_5 = keras.layers.Conv2D(
            filters=self.configs["num_hidden_units"] // 16,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation=tf.nn.leaky_relu,
            kernel_initializer=self.init,
        )

        self.decoded_final = keras.layers.Conv2D(
            filters=self.configs["n_channel"],
            kernel_size=(9, 1),
            strides=1,
            padding="valid",
            activation=None,
            kernel_initializer=self.init,
        )

    def call(self, inputs, training=None):
        # encoded = tf.cond(
        #     self.is_code_input, lambda: self.code_input, lambda: self.code_sample
        # )
        x = self.decoded_1(inputs)
        x = keras.layers.Reshape((1, 1, self.configs["num_hidden_units"]))(x)
        x = self.decoded_2(x)
        x = keras.layers.Reshape((4, 1, self.configs["num_hidden_units"] // 4))(x)
        x = self.decoded_3(x)
        x = tf.nn.depth_to_space(input=x, block_size=2)
        x = keras.layers.Reshape((8, 1, self.configs["num_hidden_units"] // 8))(x)
        x = self.decoded_4(x)
        x = tf.nn.depth_to_space(input=x, block_size=2)
        x = keras.layers.Reshape((16, 1, self.configs["num_hidden_units"] // 16))(x)
        x = self.decoded_5(x)
        x = tf.nn.depth_to_space(input=x, block_size=2)
        x = keras.layers.Reshape((self.configs["num_hidden_units"] // 16, 1, 16))(x)

        x = self.decoded_final(x)
        decoded = keras.layers.Reshape(
            (self.configs["l_win"], self.configs["n_channel"])
        )(x)

        return decoded

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({"configs": self.configs})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VAEModel(keras.Model):
    def __init__(
        self,
        configs,
        encoder: Encoder,
        decoder: Decoder,
        **kwargs,
    ):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.configs = configs
        sigma = tf.cast(self.configs["sigma"], tf.float32)
        self.sigma2 = tf.square(sigma)
        self.input_dims = self.configs["l_win"] * self.configs["n_channel"]
        self.epoch_loss_tracker = keras.metrics.Mean(name="elbo_loss")

    def call(self, inputs, training=None, mask=None):
        print(f"VAEModel input size {inputs.shape}")
        inputs = tf.reshape(
            inputs, [-1, self.configs["l_win"], self.configs["n_channel"]]
        )
        print(f"VAEModel input size (after reshape) {inputs.shape}")
        code_sample, code_mean, code_std_dev = self.encoder(inputs)
        decoded = self.decoder(code_sample)
        return code_sample, code_mean, code_std_dev, decoded

    def get_config(self):
        config = super(VAEModel, self).get_config()
        config.update(
            {
                "configs": self.configs,
                "encoder": self.encoder.get_config(),
                "decoder": self.decoder.get_config(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        print(config)
        config = config["config"]
        config["encoder"] = Encoder.from_config(config["encoder"])
        config["decoder"] = Decoder.from_config(config["decoder"])
        return cls(**config)

    def compute_loss(self, x, y_pred):
        # 在这里定义你的损失计算逻辑
        # KL divergence loss - analytical result
        original_signal = x
        code_sample, code_mean, code_std_dev, decoded = y_pred
        KL_loss = 0.5 * (
            tf.reduce_sum(tf.square(code_mean), 1)
            + tf.reduce_sum(tf.square(code_std_dev), 1)
            - tf.reduce_sum(tf.math.log(tf.square(code_std_dev)), 1)
            - self.configs["code_size"]
        )
        KL_loss = tf.reduce_mean(KL_loss)

        # norm 1 of standard deviation of the sample-wise encoder prediction
        std_dev_norm = tf.reduce_mean(code_std_dev, axis=0)

        weighted_reconstruction_error_dataset = tf.reduce_sum(
            tf.square(original_signal - decoded), [1, 2]
        )
        weighted_reconstruction_error_dataset = tf.reduce_mean(
            weighted_reconstruction_error_dataset
        )
        weighted_reconstruction_error_dataset = (
            weighted_reconstruction_error_dataset / (2 * self.sigma2)
        )

        # least squared reconstruction error
        ls_reconstruction_error = tf.reduce_sum(
            tf.square(original_signal - decoded), [1, 2]
        )
        ls_reconstruction_error = tf.reduce_mean(ls_reconstruction_error)

        # sigma regularisor - input elbo
        sigma_regularisor_dataset = self.input_dims / 2 * tf.math.log(self.sigma2)
        two_pi = self.input_dims / 2 * tf.constant(2 * np.pi)

        elbo_loss = (
            two_pi
            + sigma_regularisor_dataset
            + 0.5 * weighted_reconstruction_error_dataset
            + KL_loss
        )

        return (
            elbo_loss,
            two_pi,
            sigma_regularisor_dataset,
            KL_loss,
            weighted_reconstruction_error_dataset,
            ls_reconstruction_error,
            std_dev_norm,
        )

    def train_step(self, data):
        x = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y_pred)[0]
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.epoch_loss_tracker.update_state(loss)
        return {"loss": self.epoch_loss_tracker.result()}

    def test_step(self, data):
        x = tf.cast(data, tf.float32)
        y_pred = self(x, training=False)
        loss = self.compute_loss(x, y_pred)[0]
        self.epoch_loss_tracker.update_state(loss)
        # plot the original and reconstructed signals
        return {"loss": self.epoch_loss_tracker.result()}

    def predict_step(self, data):
        data = tf.cast(data, tf.float32)
        result = super().predict_step(data)
        return (data, result)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.epoch_loss_tracker]


class LSTMModule(keras.Layer):
    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)
        self.configs = configs
        self.lstm_1 = keras.layers.LSTM(
            units=self.configs["num_hidden_units_lstm"],
            return_sequences=True,
            return_state=False,
        )
        self.lstm_2 = keras.layers.LSTM(
            units=self.configs["num_hidden_units_lstm"],
            return_sequences=True,
            return_state=False,
        )
        self.lstm_output = keras.layers.LSTM(
            units=self.configs["code_size"],
            return_sequences=True,
            return_state=False,
        )

    def call(self, inputs, training=None, mask=None):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)
        x = self.lstm_output(x)
        return x

    def get_config(self):
        config = super(LSTMModule, self).get_config()
        config.update({"configs": self.configs})
        return config

    @classmethod
    def from_config(cls, config):
        config = config["config"]
        return cls(**config)


class VAE_LSTM(keras.Model):
    def __init__(
        self,
        configs,
        vae: VAEModel,
        lstm: LSTMModule,
        train_writer=None,
        mean: float = None,
        std: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.configs = configs
        self.vae = vae
        self.lstm = lstm
        self.vae.trainable = False
        self.epoch_loss_tracker = keras.metrics.Mean(name="mse_loss")
        # self.train_writer = train_writer
        self.mean = mean
        self.std = std

    def call(self, inputs, training=None, mask=None):
        result = self.vae(inputs)
        code_sample, code_mean, code_std_dev, decoded = result
        lstm_input = tf.reshape(
            code_mean, [-1, self.configs["l_seq"], self.configs["code_size"]]
        )
        x = self.lstm(lstm_input[:, :-1, :])
        return x, lstm_input

    def train_step(self, data):
        x = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            y_pred, y = self(x, training=True)
            y = y[:, :-1, :]
            loss = self.compute_loss(y=y, y_pred=y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.epoch_loss_tracker.update_state(loss)
        # with self.train_writer.as_default(step=self._step_counter):
        #     tf.summary.scalar("mse", loss)
        return {
            "mean_mse": self.epoch_loss_tracker.result(),
            "mse": loss,
        }

    def test_step(self, data):
        x = tf.cast(data, tf.float32)
        y_pred, y = self(x, training=False)
        y = y[:, :-1, :]
        loss = self.compute_loss(y=y, y_pred=y_pred)
        self.epoch_loss_tracker.update_state(loss)
        return {"mean_mse": self.epoch_loss_tracker.result()}

    def predict_step(self, data):
        data = tf.cast(data, tf.float32)
        code_pred, code_vae = super().predict_step(data)

        decoder_input = tf.reshape(code_pred, (-1, self.configs["code_size"]))
        decoder_input_vae = tf.reshape(code_vae, (-1, self.configs["code_size"]))

        predicted_signal = self.vae.decoder(
            decoder_input
        )  # use code mean to predict signal
        predicted_signal_vae = self.vae.decoder(decoder_input_vae)

        return (
            code_pred,
            code_vae,
            predicted_signal,
            predicted_signal_vae,
            data,
        )

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.epoch_loss_tracker]

    def get_config(self):
        config = super(VAE_LSTM, self).get_config()
        config.update(
            {
                "configs": self.configs,
                "vae": self.vae,
                "lstm": self.lstm,
                "mean": self.mean,
                "std": self.std,
                # Include other parameters needed to recreate the model
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        print(config)
        config["vae"] = VAEModel.from_config(config["vae"], custom_objects)
        config["lstm"] = LSTMModule.from_config(config["lstm"])
        return cls(**config)


class ELBOLoss(keras.losses.Loss):
    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=None):
        super().__init__(name, reduction, dtype)

    def call(self, y_true, y_pred):
        return y_pred.elbo_loss
