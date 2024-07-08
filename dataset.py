import tensorflow as tf
import pandas as pd
import numpy as np


class NSV_TFDataset(tf.data.Dataset):
    def _generator(
        csv_file,
        time_column,
        features,
        len_series,
        esample_dua,
        time_format,
        only_anomaly,
        only_normal,
    ):
        # 读取数据
        raw_data = pd.read_csv(csv_file.decode("utf-8"))
        time_column = time_column.decode("utf-8")
        time_format = time_format.decode("utf-8")
        esample_dua = esample_dua.decode("utf-8") if esample_dua is not None else None
        features = list([f.decode("utf-8") for f in features])
        assert "NSV" in raw_data.columns, "NSV not in columns"
        assert time_column in raw_data.columns, f"{time_column} not in columns"
        assert all(
            [f in raw_data.columns for f in features]
        ), f"{features} not in columns"

        raw_data[time_column] = pd.to_datetime(
            raw_data[time_column], format=time_format
        )
        raw_data.set_index(time_column, inplace=True)
        raw_data.sort_index(inplace=True)

        if esample_dua is not None:
            raw_data = raw_data.resample(esample_dua).mean().interpolate()

        raw_data["anomaly"] = raw_data["NSV"] >= 1.1

        means = {}
        stds = {}

        # 归一化
        for feature in features:
            if feature == "NSV":
                continue
            raw_data[feature] = (
                raw_data[feature] - raw_data[feature].mean()
            ) / raw_data[feature].std()
            means[feature] = raw_data[feature].mean()
            stds[feature] = raw_data[feature].std()

        raw_data["NSV"] = (raw_data["NSV"] - raw_data["NSV"].mean()) / raw_data[
            "NSV"
        ].std()
        means["NSV"] = raw_data["NSV"].mean()
        stds["NSV"] = raw_data["NSV"].std()

        if only_anomaly:
            raw_data = raw_data[raw_data["anomaly"]]
        elif only_normal:
            raw_data = raw_data[~raw_data["anomaly"]]

        # 生成数据
        for idx in range(len(raw_data) - len_series):
            start_idx = idx
            end_idx = idx + len_series

            labels = raw_data["anomaly"].iloc[start_idx:end_idx].values
            features_array = raw_data[features].iloc[start_idx:end_idx].values
            nsvs = raw_data["NSV"].iloc[start_idx:end_idx].values
            indices = start_idx
            timestamps = (
                raw_data.index[start_idx].to_pydatetime().timestamp(),
                raw_data.index[end_idx - 1].to_pydatetime().timestamp(),
            )

            yield (features_array, labels, nsvs, indices, timestamps)

    def __new__(
        cls,
        csv_file,
        time_column,
        features,
        len_series,
        esample_dua=None,
        time_format="%Y/%m/%d %H:%M",
        only_anomaly=False,
        only_normal=False,
    ):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(
                csv_file,
                time_column,
                features,
                len_series,
                esample_dua,
                time_format,
                only_anomaly,
                only_normal,
            ),
            output_signature=(
                tf.TensorSpec(shape=(len_series, len(features)), dtype=tf.float32),
                tf.TensorSpec(shape=(len_series,), dtype=tf.uint8),
                tf.TensorSpec(shape=(len_series,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(2,), dtype=tf.float32),
            ),
        )


if __name__ == "__main__":
    dataset = NSV_TFDataset(
        "Case4_B-302K-left-all-f-original-ts.csv",
        "TIMESTAMP",
        [
            "NSV",
        ],
        3,
        esample_dua="1h",
        only_anomaly=False,
        only_normal=False,
    )

    for data in dataset:
        print(data)
        break
