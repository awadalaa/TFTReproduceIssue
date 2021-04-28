import os
import itertools
import pprint
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from data import coders
from typing import Dict
from typing import List
from abc import ABC, abstractmethod
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
import shutil

import json

class RawFeatureSchema(ABC):
    def get_int_feature_keys(self) -> List[str]:
        return ['age', 'fnlwgt', 'education-num',
                'capital-gain', 'capital-loss', 'hours-per-week', ]

    def get_categorical_feature_keys(self) -> List[str]:
        return ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex',
                'native-country', ]

    def get_label_keys(self) -> List[str]:
        return ['label']

    def get_raw_feature_keys(self):
        return {
            "label_keys": self.get_label_keys(),
            "int_keys": self.get_int_feature_keys(),
            "categorical_keys": self.get_categorical_feature_keys(),
        }

    def get_raw_feature_spec(self, remove_label_keys=True):
        feature_keys = self.get_raw_feature_keys()
        feature_spec = dict(
            [(name, tf.io.VarLenFeature(tf.int64))
             for name in feature_keys["int_keys"]] +
            [(name, tf.io.VarLenFeature(tf.string))
             for name in feature_keys["categorical_keys"]] +
            [(name, tf.io.VarLenFeature(tf.string))
             for name in feature_keys["label_keys"]]
        )
        if remove_label_keys:
            label_keys = self.get_label_keys()
            for label in label_keys:
                feature_spec.pop(label)
        return feature_spec

class TFTRunner(object):
    def get_tft_feature_schema(self) -> RawFeatureSchema:
        return RawFeatureSchema()

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

    def run(self):
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

class TFTJob(TFTRunner):
    def impute(self, feature_tensor: tf.Tensor, default) -> tf.Tensor:
      sparse = tf.sparse.SparseTensor(
          feature_tensor.indices,
          feature_tensor.values,
          [feature_tensor.dense_shape[0], 1],
      )
      dense = tf.sparse.to_dense(sp_input=sparse, default_value=default)

      return tf.squeeze(dense, axis=1)


    def set_defaults(self, feature_tensor):
        if feature_tensor.dtype == tf.string:
            return self.impute(feature_tensor, "")
        elif feature_tensor.dtype == tf.float32:
            return self.impute(feature_tensor, 0.0)
        else:
            return self.impute(feature_tensor, 0)

    def get_tfidf(self, feature_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        outputs = dict()
        VOCAB_SIZE = 100000
        DELIMITERS = ".,!?() "
        for key, feature in feature_dict.items():
            word_tokens = tf.compat.v1.string_split(feature, DELIMITERS)
            word_indices = tft.compute_and_apply_vocabulary(
                word_tokens, top_k=VOCAB_SIZE, vocab_filename=f"vocab_compute_and_apply_{key}"
            )
            bow_indices, tfidf_weight = tft.tfidf(word_indices, VOCAB_SIZE + 1)
            tfidf_score = tf.math.reduce_mean(tf.sparse.to_dense(tfidf_weight), axis=-1)
            outputs[f"{key}_tfidf_score"] = tf.where(
                tf.math.is_nan(tfidf_score), tf.zeros_like(tfidf_score), tfidf_score
            )
        return outputs

    def preprocessing_fn(self, inputs):
        outputs = {}
        schema = self.get_tft_feature_schema()
        feature_keys = schema.get_raw_feature_keys()

        for key in feature_keys["int_keys"]:
            raw_feature = self.impute(inputs[key], 0)
            outputs[key] = tft.scale_to_0_1(raw_feature)

        for key in feature_keys["categorical_keys"]:
            raw_feature = self.impute(inputs[key], "")
            tft.vocabulary(raw_feature, vocab_filename=key)
            outputs[key] = raw_feature

        inputs_tfidf = {
            key: self.impute(inputs[key], "")
            for key in ['education', 'marital-status']
        }
        outputs_tfidf = self.get_tfidf(inputs_tfidf)

        for key in feature_keys["label_keys"]:
            raw_value = self.impute(inputs[key], "")
            # For the label column we provide the mapping from string to index.
            table_keys = ['>50K', '<=50K']
            initializer = tf.lookup.KeyValueTensorInitializer(
                keys=table_keys,
                values=tf.cast(tf.range(len(table_keys)), tf.int64),
                key_dtype=tf.string,
                value_dtype=tf.int64)
            table = tf.lookup.StaticHashTable(initializer, default_value=-1)
            # Remove trailing periods for test data when the data is read with tf.data.
            label_str = tf.strings.regex_replace(raw_value, r'\.', '')
            label_str = tf.strings.strip(label_str)
            data_labels = table.lookup(label_str)
            transformed_label = tf.one_hot(
                indices=data_labels, depth=len(table_keys), on_value=1.0, off_value=0.0)
            outputs[key] = tf.reshape(transformed_label, [-1, len(table_keys)])


        all_outputs = {
            **outputs,
            **outputs_tfidf
        }

        return all_outputs