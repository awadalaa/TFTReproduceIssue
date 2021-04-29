import os
import pprint
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_addons as tfa

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
            batch_size=train_batch_size
        )
        self.eval_steps = num_eval_examples // eval_batch_size
        self.eval_dataset = self.get_dataset_batch(
            file_pattern=eval_data_pattern,
            tft_transform_output=self.tft_transform_output,
            batch_size=eval_batch_size,
            limit=num_eval_examples
        )

    def get_dataset_batch(
            self,
            file_pattern: str,
            tft_transform_output: tft.TFTransformOutput,
            batch_size: int,
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

    def attach_prediction_head(self, model):
        probabilities = tf.keras.layers.Activation("sigmoid", name="predictions")(
            model.output
        )
        return tf.keras.Model(inputs=model.inputs, outputs=probabilities)

    def export_model(self, model):
        feature_forward_keys = []
        full_model = self.attach_prediction_head(model)

        full_model.save(
            filepath=self.model_dir,
            overwrite=True,
            signatures=self.get_serve_tf_examples_fn(model, feature_forward_keys),
        )


    def get_serve_tf_examples_fn(self, model, forwarding_keys=[]):
        # Returns a function that parses a serialized tf.Example and applies TFT.
        model.tft_layer = self.tft_transform_output.transform_features_layer()

        @tf.function
        def extract_forwarded_features(raw_features):
            forwarded_features = {}
            for key in forwarding_keys:
                if key not in raw_features:
                    raise ValueError(
                        "Forwarded feature {} does not exist! Available features: {}".format(
                            key, [*raw_features.keys()]
                        )
                    )

                feature = raw_features[key]
                with tf.name_scope("forward_features"):
                    # Export signatures only take dense tensors
                    if isinstance(feature, tf.SparseTensor):
                        feature = tf.sparse.to_dense(feature, name="sparse_to_dense")

                # Keeping the export signature for forwarded features the same as the Estimator API
                forwarded_features[key] = tf.squeeze(feature, axis=-1)

            return forwarded_features

        @tf.function
        def inference_model(serialized_tf_examples):
            # Returns the output to be used in the serving signature.
            raw_feature_spec = self.tft_transform_output.raw_feature_spec()
            raw_feature_spec.pop(self.get_label_key())

            parsed_features = tf.io.parse_example(
                serialized_tf_examples, raw_feature_spec
            )

            transformed_features = model.tft_layer(parsed_features)
            forwarded_features = extract_forwarded_features(parsed_features)
            return model(transformed_features), forwarded_features

        @tf.function
        def serving_default_signature(serialized_examples):
            logits, forwarded_features = inference_model(serialized_examples)
            two_class_logits = tf.concat(
                (tf.zeros_like(logits), logits), axis=-1, name="two_class_logits"
            )

            return {
                "scores": tf.keras.layers.Softmax(name="probabilities")(
                    two_class_logits
                ),
                **forwarded_features,
            }

        @tf.function
        def predict_signature(serialized_examples):
            logits, forwarded_features = inference_model(serialized_examples)
            two_class_logits = tf.concat(
                (tf.zeros_like(logits), logits), axis=-1, name="two_class_logits"
            )

            return {
                "logits": logits,
                "logistic": tf.keras.layers.Activation("sigmoid")(logits),
                "probabilities": tf.keras.layers.Softmax(name="probabilities")(
                    two_class_logits
                ),
                **forwarded_features,
            }

        return {
            "serving_default": serving_default_signature.get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="inputs")
            ),
            "predict": predict_signature.get_concrete_function(
                tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
            ),
        }

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
        output = tf.keras.layers.Dense(50, activation='relu')(output)
        output = tf.keras.layers.Dense(20, activation='relu')(output)
        output = tf.keras.layers.Dense(1, name="logits")(output)
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
                optimizer=self.get_optimizer(),
                loss=self.get_loss(),
                metrics=self.get_metrics(),
            )
        return model

    def get_optimizer(self):
        return tfa.optimizers.LazyAdam(
            learning_rate=0.01,
            epsilon=1e-08,
            beta_1=0.9,
            beta_2=0.999,
        )

    def get_loss(self):
        return tf.keras.losses.BinaryCrossentropy()

    def get_metrics(self):
        return [
            tf.keras.metrics.BinaryAccuracy(),
        ]

    def train_and_evaluate(self):
        model = self.get_compiled_model()
        model.summary()

        for x, y in self.train_dataset.take(1):
            print("TRANSORMED FEATURES:")
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