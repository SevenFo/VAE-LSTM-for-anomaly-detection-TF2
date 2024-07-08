import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import datetime
import pandas as pd

from models import VAEModel, Encoder, Decoder, VAE_LSTM, LSTMModule
from callbacks import VAELSTMModelCallback, VAEModelCallback, TensorBoardFix
from utils import load_csv, post_process

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {
        "exp_name": "NAB",
        "dataset": "NSV_sub_right_sub_4374",
        "y_scale": 5,
        "one_image": 0,
        "l_seq": 12,
        "l_win": 24,
        "n_channel": 1,
        "TRAIN_VAE": 1,
        "TRAIN_LSTM": 1,
        "TRAIN_sigma": 0,
        "batch_size": 32,
        "batch_size_lstm": 32,
        "load_model": 1,
        "load_dir": "default",
        "num_epochs_vae": 0,
        "num_epochs_lstm": 20,
        "learning_rate_vae": 0.0004,
        "learning_rate_lstm": 0.0002,
        "code_size": 6,
        "sigma": 0.1,
        "sigma2_offset": 0.01,
        "num_hidden_units": 512,
        "num_hidden_units_lstm": 64,
    }
    train_dataset, test_dataset, dataset_metadata = load_csv(
        csv_file="Case4_B-302K-left-all-f-original-ts.csv",
        time_column="TIMESTAMP",
        feature_cols=["NSV"],
        len_series=24,
        esample_dua="1h",
        only_normal=True,
    )
    train_dataset = (
        train_dataset.shuffle(buffer_size=1000)
        .repeat(8000)
        .batch(config["batch_size"], drop_remainder=True)
    )
    test_dataset = (
        test_dataset.batch(config["batch_size"], drop_remainder=True).take(1).repeat(1)
    )
    lstm_train_dataset, lstm_test_dataset, lstm_dataset_metadata = load_csv(
        csv_file="Case4_B-302K-left-all-f-original-ts.csv",
        time_column="TIMESTAMP",
        feature_cols=["NSV"],
        len_series=config["l_seq"] * config["l_win"],
        esample_dua="1h",
        only_normal=True,
    )
    lstm_train_dataset = (
        lstm_train_dataset.shuffle(buffer_size=1000)
        .repeat(8000)
        .batch(config["batch_size_lstm"], drop_remainder=True)
    )
    lstm_test_dataset = (
        lstm_test_dataset.batch(config["batch_size_lstm"], drop_remainder=True)
        .take(1)
        .repeat(1)
    )

    train_log_dir = "./logs/VAE-LSTM/" + current_time + "/train"
    test_log_dir = "./logs/VAE-LSTM/" + current_time + "/test"
    predict_log_dir = "./logs/VAE-LSTM/" + current_time + "/predict"
    train_writer = tf.summary.create_file_writer(train_log_dir)
    # test_writer = tf.summary.create_file_writer(test_log_dir)
    predict_writer = tf.summary.create_file_writer(predict_log_dir)

    vae_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_vae"], decay_steps=1000, decay_rate=0.96, staircase=True
    )
    lstm_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config["learning_rate_lstm"], decay_steps=1000, decay_rate=0.96, staircase=True
    )

    vae_model = VAEModel(config, encoder=Encoder(config), decoder=Decoder(config))
    vae_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=vae_lr_schedule),
        run_eagerly=False,
    )
    # 构建模型
    # 使用示例输入数据的形状来调用模型
    example_input = tf.random.normal([1, config["l_win"], config["n_channel"]])
    vae_model(example_input)
    # 打印模型总结
    vae_model.summary()

    tb_callback = TensorBoardFix("./logs/VAE-LSTM/" + current_time, update_freq=1)

    # train VAE model
    vae_model.fit(
        train_dataset,
        epochs=20,
        steps_per_epoch=1000,
        callbacks=[tb_callback],
    )
    vae_model.predict(test_dataset, callbacks=VAEModelCallback(writer=predict_writer))

    # train VAE_LSTM model
    vae_lstm = VAE_LSTM(
        config,
        vae=vae_model,
        lstm=LSTMModule(config),
        std=dataset_metadata["stds"]["NSV"],
        mean=dataset_metadata["means"]["NSV"],
    )
    vae_lstm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lstm_lr_schedule),
        loss=keras.losses.MeanSquaredError(),
        run_eagerly=False,
    )
    example_input = tf.random.normal(
        [1, config["l_seq"], config["l_win"], config["n_channel"]]
    )
    vae_lstm(example_input)
    vae_lstm.summary()
    vae_lstm.fit(
        lstm_train_dataset,
        epochs=15,
        steps_per_epoch=1200,
        callbacks=[
            tb_callback,
            VAELSTMModelCallback(writer=train_writer),
        ],
    )
    vae_lstm.predict(
        lstm_test_dataset, callbacks=VAELSTMModelCallback(writer=predict_writer)
    )

    vae_lstm.save("VAE_LSTM.keras")
    del train_dataset, test_dataset, lstm_train_dataset, lstm_test_dataset

    dataset, _, dataset_metadata = load_csv(
        csv_file="Case4_B-302K-left-all-f-original-ts.csv",
        time_column="TIMESTAMP",
        feature_cols=["NSV"],
        len_series=config["l_seq"] * config["l_win"],
        esample_dua="1h",
        only_normal=False,
        split=None,
        concise=False,
    )
    dataset = dataset.batch(2048)
    score_frames = []  # 创建一个空列表来收集所有要添加的DataFrame
    for batch in dataset:
        features = batch["features"]
        labels = batch["labels"]
        concise_labels = batch["concise_labels"]
        # indecies = batch["indecies"]
        print(f"input batch size: {features.shape}")
        seq_indecies = batch["seq_indecies"]

        code_pred, code_vae, predicted_signal, predicted_signal_vae, original_signal = (
            vae_lstm.predict(features, batch_size=features.shape[0])
        )

        (
            predicted_signal,
            predicted_signal_vae,
            original_signal,
            scores,
            anomaly_indices,
        ) = post_process(
            vae_lstm.mean,
            vae_lstm.std,
            predicted_signal,
            predicted_signal_vae,
            original_signal,
            configs=config,
        )

        score_frames.append(
            pd.DataFrame(
                {
                    "idx": seq_indecies,
                    "label": concise_labels,
                    "score": scores,
                }
            )
        )
    score_frame = pd.concat(score_frames)
    score_frame.to_csv("score_frame.csv")
