import tensorflow as tf
import tensorflow_addons as tfa
import pprint

class KerasPredictTest(object):
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def generate_example(self):
        features = {
            'workclass': self._bytes_feature(b"Private"),
            'education': self._bytes_feature(b"Bachelors"),
            'marital-status': self._bytes_feature(b"Divorced"),
            'occupation': self._bytes_feature(b"Tech-support"),
            'relationship': self._bytes_feature(b"Unmarried"),
            'race': self._bytes_feature(b"Other"),
            'sex': self._bytes_feature(b"Male"),
            'native-country': self._bytes_feature(b"United-States"),
            'age': self._int64_feature(30),
            'capital-gain': self._int64_feature(215646),
            'capital-loss': self._int64_feature(2174),
            'hours-per-week': self._int64_feature(40),
            'education-num': self._int64_feature(13),
        }

        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto

    # use this with model(dataset.map(prepare))
    def prepare(self, record, feature_keys):
        model_inputs = [[record[ft]] for ft in feature_keys]
        return model_inputs

    # use this with model.signatures['serving_default'](dataset.map(prepare_serving))
    def prepare_serving(self, record, feature_keys):
        model_inputs = {ft: record[ft] for ft in feature_keys}
        return model_inputs

