import tensorflow as tf
import tensorflow.python.keras as keras
from keras.api.callbacks import Callback
import matplotlib.pyplot as plt

from utils import plot_to_image, post_process


class VAEModelCallback(Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        # keys = list(logs.keys())
        # print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
        data, results = logs.get("outputs")
        code_sample, code_mean, code_std_dev, decoded = results
        fig, axies = plt.subplots(8, 4, figsize=(16, 20))
        axies = axies.flatten()
        for i in range(32):
            axies[i].plot(data[i])
            axies[i].plot(decoded[i])
            axies[i].set_title(f"Sample {i}")
            axies[i].legend(["Original", "Predicted"])
        fig.tight_layout()
        # save figure to tensorboard
        with self.writer.as_default():
            tf.summary.image("Original VS Predicted", plot_to_image(fig), step=batch)


class VAELSTMModelCallback(Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def on_predict_batch_end(self, batch, logs=None):
        # TODO Hardcoded shape of the data
        (
            code_pred,  # shorter one time step
            code_vae,
            predicted_signal,  # same length as original, as first time step has been fiiled with original
            predicted_signal_vae,
            original_signal,
        ) = logs.get("outputs")

        (
            predicted_signal,
            predicted_signal_vae,
            original_signal,
            scores,
            anomaly_indices,
        ) = post_process(
            self.model.mean,
            self.model.std,
            predicted_signal,
            predicted_signal_vae,
            original_signal,
            configs=self.model.configs,
        )

        # plot code mean original vs predicted
        code_vae = tf.reshape(
            code_vae, (-1, self.model.configs["l_seq"], self.model.configs["code_size"])
        )
        code_pred = tf.reshape(
            code_pred,
            (-1, self.model.configs["l_seq"] - 1, self.model.configs["code_size"]),
        )
        fig, axes = plt.subplots(8, 4, figsize=(16, 20))
        axes = axes.flatten()
        fig.suptitle("Code-Mean Original VS Predicted")  # 设置总标题
        for i in range(32):
            # only plot the last time step
            axes[i].plot(code_vae[i][-1], label="Original")
            axes[i].plot(code_pred[i][-1], label="Predicted")
            axes[i].set_title(f"Sample {i}")
            axes[i].legend()
        fig.tight_layout()
        with self.writer.as_default():
            tf.summary.image(
                "Code-Mean Original VS Predicted", plot_to_image(fig), step=batch
            )

        original_signal_short = tf.reshape(
            original_signal,
            (32, self.model.configs["l_seq"], self.model.configs["l_win"]),
        )
        predicted_signal_short = tf.reshape(
            predicted_signal,
            (32, self.model.configs["l_seq"], self.model.configs["l_win"]),
        )
        predicted_signal_vae_short = tf.reshape(
            predicted_signal_vae,
            (32, self.model.configs["l_seq"], self.model.configs["l_win"]),
        )

        # plot signal original vs predicted

        fig, axes = plt.subplots(8, 4, figsize=(16, 20))  # figsize=(width, height)
        axes = axes.flatten()
        for i in range(32):
            # only plot the last time step
            axes[i].plot(original_signal_short[i][-1], label="Original")
            axes[i].plot(predicted_signal_short[i][-1], label="Predicted")
            axes[i].set_title(f"Sample {i}")
            axes[i].legend()
        fig.suptitle("LSTM Signal Original VS Predicted")
        fig.tight_layout()
        with self.writer.as_default():
            tf.summary.image(
                "LSTM Signal Original VS Predicte", plot_to_image(fig), step=batch
            )

        fig_vae, axes_vae = plt.subplots(8, 4, figsize=(16, 20))
        axes_vae = axes_vae.flatten()
        for i in range(32):
            axes_vae[i].plot(original_signal_short[i][-1], label="Original")
            axes_vae[i].plot(predicted_signal_vae_short[i][-1], label="Predicted")
            axes_vae[i].set_title(f"Sample {i}")
            axes_vae[i].legend()
        fig_vae.suptitle("VAE Signal Original VS Predicted")
        fig_vae.tight_layout()
        with self.writer.as_default():
            tf.summary.image(
                "VAE Signal Original VS Predicted", plot_to_image(fig_vae), step=batch
            )

        # plot long signal original vs predicted

        fig_long, axes_long = plt.subplots(16, 2, figsize=(16, 20 * 2))
        axes_long = axes_long.flatten()
        for i in range(32):
            axes_long[i].plot(original_signal[i], label="Original")
            axes_long[i].plot(
                predicted_signal[i],
                label="Predicted",
            )
            axes_long[i].set_title(f"Sample {i}")
            axes_long[i].legend()
        fig_long.suptitle("LSTM-Signal-Long-Original-VS-Predicted")
        fig_long.tight_layout()
        with self.writer.as_default():
            tf.summary.image(
                "LSTM-Signal-Long-Original-VS-Predicted",
                plot_to_image(fig_long),
                step=batch,
            )

        fig_vae_long, axes_vae_long = plt.subplots(16, 2, figsize=(16, 20 * 2))
        axes_vae_long = axes_vae_long.flatten()
        for i in range(32):
            axes_vae_long[i].plot(original_signal[i], label="Original")
            axes_vae_long[i].plot(predicted_signal_vae[i], label="Predicted")
            axes_vae_long[i].set_title(f"Sample {i}")
            axes_vae_long[i].legend()
        fig_vae_long.suptitle("VAE-Signal-Long-Original-VS-Predicted")
        fig_vae_long.tight_layout()
        with self.writer.as_default():
            tf.summary.image(
                "VAE-Signal-Long-Original-VS-Predicted",
                plot_to_image(fig_vae_long),
                step=batch,
            )


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    """

    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)

    def on_train_batch_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_batch_begin(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_batch_end(*args, **kwargs)
        self._train_step += 1

    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)
