import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict
from data.base_runner import BaseTFTRunner, BaseRawFeatureSchema
from data.raw_features import RawFeatureSchema

class TFTJob(BaseTFTRunner):
    def get_tft_feature_schema(self) -> BaseRawFeatureSchema:
        return RawFeatureSchema()

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