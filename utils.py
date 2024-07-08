from typing import Union
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas
import numpy as np


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def load_csv(
    csv_file: str,
    time_column: str,
    feature_cols: list,
    len_series: int,
    esample_dua: Union[str, None] = None,
    time_format: str = "%Y/%m/%d %H:%M",
    only_normal: bool = False,
    split: float = 0.8,
    concise: bool = True,
):
    n_channels = len(feature_cols)
    raw_data = pandas.read_csv(csv_file)

    assert "NSV" in raw_data.columns, "NSV not in columns"
    assert time_column in raw_data.columns, f"{time_column} not in columns"
    assert all(
        [f in raw_data.columns for f in feature_cols]
    ), f"{feature_cols} not in columns"

    raw_data[time_column] = pandas.to_datetime(
        raw_data[time_column], format=time_format
    )
    raw_data.set_index(time_column, inplace=True)
    raw_data.sort_index(inplace=True)

    if esample_dua is not None:
        raw_data = raw_data.resample(esample_dua).mean().interpolate()

    raw_data["anomaly"] = raw_data["NSV"] >= 1.1

    means = {}
    stds = {}

    # normalize
    for feature in feature_cols:
        if feature == "NSV":
            continue
        raw_data[feature] = (raw_data[feature] - raw_data[feature].mean()) / raw_data[
            feature
        ].std()
        means[feature] = raw_data[feature].mean()
        stds[feature] = raw_data[feature].std()

    means["NSV"] = raw_data["NSV"].mean()
    stds["NSV"] = raw_data["NSV"].std()

    raw_data["NSV"] = (raw_data["NSV"] - means["NSV"]) / stds["NSV"]

    timestampes = []
    nsvs = []
    datas = []
    raw_data = raw_data

    if concise:
        for idx in range(len(raw_data) - len_series):
            if only_normal and any(raw_data["anomaly"].iloc[idx : idx + len_series]):
                continue
            datas.append(raw_data[feature_cols].iloc[idx : idx + len_series].values)
        # 将feature_values转换为NumPy数组
        datas = np.array(datas)

        random_indices = np.random.permutation(len(datas)).astype(int)
        train_indices = (
            random_indices[: int(len(datas) * split)] if split else random_indices
        )
        test_indices = random_indices[int(len(datas) * split) :] if split else None
        train_data = datas[train_indices]
        test_data = datas[test_indices]

    else:
        features = []
        labels = []
        concise_labels = []
        seq_indecies = []
        for idx in range(len(raw_data) - len_series):
            if only_normal and any(raw_data["anomaly"].iloc[idx : idx + len_series]):
                continue
            features.append(raw_data[feature_cols].iloc[idx : idx + len_series].values)
            labels.append(raw_data["anomaly"].iloc[idx : idx + len_series].values)
            concise_labels.append(any(raw_data["anomaly"].iloc[idx : idx + len_series]))
            seq_indecies.append(idx)
        random_indices = np.random.permutation(len(features)).astype(int)
        train_indices = (
            random_indices[: int(len(features) * split)] if split else random_indices
        )
        test_indices = random_indices[int(len(features) * split) :] if split else None
        features = np.array(features)
        labels = np.array(labels)
        concise_labels = np.array(concise_labels)
        seq_indecies = np.array(seq_indecies)

        train_data = {
            "features": features[train_indices],
            "labels": labels[train_indices],
            "concise_labels": concise_labels[train_indices],
            "seq_indecies": seq_indecies[train_indices],
        }
        test_data = {
            "features": features[test_indices],
            "labels": labels[test_indices],
            "concise_labels": concise_labels[test_indices],
            "seq_indecies": seq_indecies[test_indices],
        }

    return (
        tf.data.Dataset.from_tensor_slices(train_data),
        tf.data.Dataset.from_tensor_slices(test_data)
        if test_indices is not None
        else None,
        {"means": means, "stds": stds},
    )


def detect_step(y, y_pred, threshold_per: float = None):
    scores = calculate_anomaly_score(y, y_pred)
    if threshold_per is None:
        return scores, np.zeros_like(scores)
    threshold = np.percentile(scores, threshold_per * 100)
    anomaly_indices = scores > threshold
    return scores, anomaly_indices


def calculate_anomaly_score(y, y_pred):
    # calculate the anomaly score
    mse = np.mean(np.square(y - y_pred), axis=1)
    return mse


def post_process(
    mean, std, predicted_signal, predicted_signal_vae, original_signal, configs
):
    l_seq = configs["l_seq"]
    l_win = configs["l_win"]
    predicted_signal = np.reshape(predicted_signal, (-1, (l_seq - 1) * l_win))  # long
    predicted_signal_vae = np.reshape(
        predicted_signal_vae, (-1, (l_seq) * l_win)
    )  # long

    original_signal = np.reshape(original_signal, (-1, l_seq, l_win))

    # Function to concatenate the first element of original_signal with predicted_signal
    def concat_signals(i):
        return np.concatenate([original_signal[i, 0], predicted_signal[i]], axis=0)

    # Map function over each row of predicted_signal
    predicted_signal_tmp = [concat_signals(i) for i in range(predicted_signal.shape[0])]

    # Stack the results back into a tensor
    predicted_signal = np.stack(predicted_signal_tmp)

    original_signal = np.reshape(original_signal, (-1, l_seq * l_win))  # long

    original_signal = original_signal * std + mean
    predicted_signal = predicted_signal * std + mean
    predicted_signal_vae = predicted_signal_vae * std + mean

    scores, anomaly_indices = detect_step(original_signal, predicted_signal)

    return (
        predicted_signal,
        predicted_signal_vae,
        original_signal,
        scores,
        anomaly_indices,
    )


if __name__ == "__main__":
    train_dataset, test_dataset, dataset_metadata = load_csv(
        csv_file="Case4_B-302K-left-all-f-original-ts.csv",
        time_column="TIMESTAMP",
        features=["NSV"],
        len_series=24,
        esample_dua="1h",
        only_normal=True,
    )
