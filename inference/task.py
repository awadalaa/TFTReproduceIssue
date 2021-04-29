import os
import pprint
import tensorflow as tf
from inference import predict

if __name__ == '__main__':
    predictor = predict.KerasPredictTest()
    example = predictor.generate_example()
    serialized_proto = example.SerializeToString()
    model_dir = os.path.join('model','exported')
    reloaded_model = tf.keras.models.load_model(model_dir)

    predict = reloaded_model.signatures['serving_default']
    inference_data = tf.constant([serialized_proto])
    print("Prediction Output:")
    pprint.pprint(predict(inference_data))