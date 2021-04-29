from typing import List
import tensorflow as tf
from data.base_runner import BaseRawFeatureSchema

class RawFeatureSchema(BaseRawFeatureSchema):
    def get_raw_feature_spec(self, remove_label_keys=True):
        feature_keys = self.get_raw_feature_keys()

        feature_spec = dict(
            [(name, tf.io.VarLenFeature(tf.int64)) for name in feature_keys["int_keys"]]
            + [
                (name, tf.io.VarLenFeature(tf.string))
                for name in feature_keys["categorical_keys"]
            ]
            + [
                (name, tf.io.VarLenFeature(tf.string))
                for name in feature_keys["label_keys"]
            ]
        )
        if remove_label_keys:
            label_keys = self.get_label_keys()
            for label in label_keys:
                feature_spec.pop(label)

        return feature_spec

    def get_int_feature_keys(self) -> List[str]:
        return [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]

    def get_categorical_feature_keys(self) -> List[str]:
        return [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

    def get_label_keys(self) -> List[str]:
        return ["label"]

