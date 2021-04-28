import os
import pprint
import tensorflow as tf
import tensorflow_transform as tft

class CensusModel(object):
    def __init__(self,
                 working_dir='model',
                 num_epochs=3,
                 train_batch_size=32,
                 eval_batch_size=32,
                 num_eval_examples=1000):

        self.num_epochs = num_epochs
        self.model_dir = os.path.join(working_dir, 'exported')
        train_data_pattern = os.path.join(working_dir, 'train', 'part-*')
        eval_data_pattern = os.path.join(working_dir, 'test', 'part-*')
        transform_dir = os.path.join(working_dir, 'transform_output')
        self.tft_transform_output = tft.TFTransformOutput(transform_dir)
        self.train_dataset = self.get_dataset_batch(
            file_pattern=train_data_pattern,
            tft_transform_output=self.tft_transform_output,
            batch_size=train_batch_size,
            mode=tf.estimator.ModeKeys.TRAIN,
        )
        self.eval_steps = num_eval_examples // eval_batch_size
        self.eval_dataset = self.get_dataset_batch(
            file_pattern=eval_data_pattern,
            tft_transform_output=self.tft_transform_output,
            batch_size=eval_batch_size,
            mode=tf.estimator.ModeKeys.EVAL,
            limit=num_eval_examples,
        )

    def get_dataset_batch(
            self,
            file_pattern: str,
            tft_transform_output: tft.TFTransformOutput,
            batch_size: int,
            mode: tf.estimator.ModeKeys,
            limit: int = None,
    ) -> tf.data.Dataset:
        features = tft_transform_output.transformed_feature_spec()
        dataset = tf.data.Dataset.list_files(file_pattern)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if limit != None:
            dataset = dataset.take(limit)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.apply(
            tf.data.experimental.parse_example_dataset(
                features, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        )
        dataset = dataset.map(
            map_func=lambda x: (x, x.pop(self.get_label_key())),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def export_model(self, model):
        model.tft_layer = self.tft_transform_output.transform_features_layer()

        @tf.function
        def serve_tf_examples_fn(serialized_tf_examples):
            feature_spec = self.tft_transform_output.raw_feature_spec().copy()
            feature_spec.pop('label')
            parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
            transformed_features = model.tft_layer(parsed_features)
            outputs = model(transformed_features)
            classes_names = tf.constant([['0', '1']])
            classes = tf.tile(classes_names, [tf.shape(outputs)[0], 1])
            return {'classes': classes, 'scores': outputs}

        concrete_serving_fn = serve_tf_examples_fn.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='inputs'))
        signatures = {'serving_default': concrete_serving_fn}

        versioned_output_dir = os.path.join(self.model_dir, '1')
        model.save(versioned_output_dir, save_format='tf', signatures=signatures)

    def get_label_key(self):
        return "label"

    def _get_encoded_features_columns(self):
        numeric_features = [
            tf.feature_column.numeric_column(key, shape=())
            for key in ['age', 'fnlwgt', 'education-num',
                        'capital-gain', 'capital-loss', 'hours-per-week', ]
        ]
        categorical_features = [
            tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key, self.tft_transform_output.vocabulary_file_by_name(key)
                )
            )
            for key in ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex',
                        'native-country', ]
        ]
        transformed_features = [
            tf.feature_column.numeric_column('education_tfidf_score', shape=()),
            tf.feature_column.numeric_column('marital-status_tfidf_score', shape=()),
        ]
        return numeric_features + \
               categorical_features + \
               transformed_features

    def get_model(self) -> tf.keras.models.Model:
        feature_spec = self.tft_transform_output.transformed_feature_spec().copy()
        feature_spec.pop(self.get_label_key())
        feature_inputs = {
            k: tf.keras.Input(name=k, shape=v.shape, dtype=v.dtype)
            for k, v in feature_spec.items()
        }

        encoded_inputs = self._get_encoded_features_columns()
        feature_layer = tf.keras.layers.DenseFeatures(
            feature_columns=encoded_inputs
        )(feature_inputs)

        output = tf.keras.layers.Dense(100, activation='relu')(feature_layer)
        output = tf.keras.layers.Dense(70, activation='relu')(output)
        output = tf.keras.layers.Dense(50, activation='relu')(output)
        output = tf.keras.layers.Dense(20, activation='relu')(output)
        output = tf.keras.layers.Dense(2, activation='sigmoid')(output)
        model = tf.keras.Model(inputs=feature_inputs, outputs=output)
        return model

    def get_training_strategy(self):
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) == 1:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        return strategy

    def get_compiled_model(self, from_saved_model=False):
        strategy = self.get_training_strategy()
        with strategy.scope():
            model = self.get_model()
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
        return model

    def train_and_evaluate(self):
        print("AWADALAA  train_and_evaluate")
        model = self.get_compiled_model()
        print(f"AWADALAA epochs:{self.num_epochs} dataset:{self.eval_dataset} train:{self.train_dataset}")
        model.summary()
        for x, y in self.train_dataset.take(1):
            print("AWADALAA FEATURES:")
            pprint.pprint(x)
        history = model.fit(
            self.train_dataset,
            epochs=self.num_epochs,
            validation_data=self.eval_dataset,
            validation_steps=self.eval_steps,
            verbose=1,
        )
        self.export_model(model)

        return history