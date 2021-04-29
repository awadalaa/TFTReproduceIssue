import os
import pprint
import tensorflow as tf
from inference import predict

if __name__ == '__main__':
    predictor = predict.KerasPredictTest()

    # Generate TF Proto serizliaed example
    example = predictor.generate_example()
    serialized_proto = example.SerializeToString()

    # Reload the trained keras model
    model_dir = os.path.join('model','exported')
    reloaded_model = tf.keras.models.load_model(model_dir)

    # Pass the serialized example to the model's serving function
    predict = reloaded_model.signatures['serving_default']
    inference_data = tf.constant([serialized_proto])

    print("Prediction Output:")
    pprint.pprint(predict(inference_data))