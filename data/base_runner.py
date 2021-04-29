import os
import itertools
import apache_beam as beam
from typing import List
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from abc import ABC, abstractmethod
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from data import coders


class BaseRawFeatureSchema(ABC):
    @abstractmethod
    def get_int_feature_keys(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_categorical_feature_keys(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_label_keys(self) -> List[str]:
        raise NotImplementedError

    def get_raw_feature_keys(self):
        return {
            "label_keys": self.get_label_keys(),
            "int_keys": self.get_int_feature_keys(),
            "categorical_keys": self.get_categorical_feature_keys(),
        }

    def get_raw_feature_spec(self, remove_label_keys=True):
        feature_keys = self.get_raw_feature_keys()

        feature_spec = dict(
            [(name, tf.io.VarLenFeature(tf.int64)) for name in feature_keys["int_keys"]]
            + [
                (name, tf.io.VarLenFeature(tf.string))
                for name in feature_keys["categorical_keys"]
            ]
            + [
                (name, tf.io.FixedLenFeature([], tf.float32))
                for name in feature_keys["label_keys"]
            ]
        )

        if remove_label_keys:
            label_keys = self.get_label_keys()
            for label in label_keys:
                feature_spec.pop(label)

        return feature_spec


class BaseTFTRunner(ABC):
    @abstractmethod
    def get_tft_feature_schema(self) -> BaseRawFeatureSchema:
        raise NotImplementedError

    @abstractmethod
    def preprocessing_fn(self, inputs):
        raise NotImplementedError

    def prepare_for_transform(self, raw_input_dictionary):
        outputs = {}
        raw_feature_keys = self.get_tft_feature_schema().get_raw_feature_keys()
        label_keys = raw_feature_keys.pop("label_keys")
        fields = itertools.chain.from_iterable(raw_feature_keys.values())
        for field_name in fields:
            value = raw_input_dictionary.get(field_name, None)
            if value == None:
                outputs[field_name] = []
            else:
                outputs[field_name] = [value]

        for feat in raw_feature_keys["int_keys"]:
            outputs[feat] = int(raw_input_dictionary.get(feat, 0))

        for label in label_keys:
            outputs[label] = raw_input_dictionary.get(label, "")
        return outputs

    def run(self, p_options):
        output_dir = 'model'
        input_train_file = './data/train.json'
        input_test_file = './data/test.json'

        pipeline = beam.Pipeline()
        with tft_beam.Context(
                temp_dir=os.path.join(output_dir, 'tmp'),
                force_tf_compat_v1=None):
            raw_features_schema = self.get_tft_feature_schema()
            raw_data_metadata = dataset_metadata.DatasetMetadata(
                schema_utils.schema_from_feature_spec(
                    raw_features_schema.get_raw_feature_spec(remove_label_keys=False)
                )
            )

            raw_train_data = (
                    pipeline
                    | "ReadTrainInputs" >> beam.io.ReadFromText(input_train_file, coder=coders.JsonCoder())
                    | "PreFormatTrainData" >> beam.Map(self.prepare_for_transform)
            )
            _ = raw_data_metadata | "WriteRawMetadata" >> tft_beam.WriteMetadata(
                path=os.path.join(output_dir, "transform_output", "metadata"),
                pipeline=pipeline,
            )
            raw_train_dataset = (raw_train_data, raw_data_metadata)

            transformed_train_dataset, transform_fn = (
                    raw_train_dataset
                    | tft_beam.AnalyzeAndTransformDataset(self.preprocessing_fn)
            )
            transformed_train_data, transformed_metadata = transformed_train_dataset
            transformed_data_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema
            )

            train_output_directory = os.path.join(output_dir, "train", "part")
            _ = (
                    transformed_train_data
                    | "EncodeTrainData" >> beam.Map(transformed_data_coder.encode)
                    | "WriteTrainData" >> beam.io.tfrecordio.WriteToTFRecord(
                train_output_directory, file_name_suffix=".tfrecord"
            )
            )

            raw_test_data = (
                    pipeline
                    | "ReadTestInputs" >> beam.io.ReadFromText(input_test_file, coder=coders.JsonCoder())
                    | "PreFormatTestData" >> beam.Map(self.prepare_for_transform)
            )

            raw_test_dataset = (raw_test_data, raw_data_metadata)
            transformed_test_data, _ = (
                                           raw_test_dataset,
                                           transform_fn,
                                       ) | "TransformTestDataset" >> tft_beam.TransformDataset()

            test_output_directory = os.path.join(
                output_dir, "test", "part"
            )

            _ = (
                    transformed_test_data
                    | "EncodeTestData" >> beam.Map(transformed_data_coder.encode)
                    | "WriteTestData"
                    >> beam.io.tfrecordio.WriteToTFRecord(
                test_output_directory, file_name_suffix=".tfrecord"
            )
            )
            transform_fn_output_dir = os.path.join(output_dir, "transform_output")
            _ = transform_fn | "WriteTransformFn" >> tft_beam.WriteTransformFn(
                transform_fn_output_dir
            )

            pipeline.run()