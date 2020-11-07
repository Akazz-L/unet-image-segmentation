from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
import os
import textwrap
import json
import numpy as np
from model import *

MODEL_FILENAME = "unet_membrane20201102-212716.hdf5"


class TensorFlowInferenceHandler(default_inference_handler.DefaultInferenceHandler):

    def default_model_fn(self, model_dir):
        model_path = os.path.join(model_dir, MODEL_FILENAME)  
        if not os.path.exists(model_path):
            raise FileNotFoundError("Failed to load model with default model_fn: missing file {}.".format(model_path))
        model = unet_model(pretrained_weights = model_path)
        return model


    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into Numpy Array (Batch_size, width, height, channel)for tensorflow model
        """
        # decoding
        input_data = _json_to_numpy(input_data)
        
        input_data = np.reshape(input_data,(1,input_data.shape[0],input_data.shape[1]))

        # rescaling
        input_data = input_data / 255 
        return input_data

    def default_predict_fn(self, input_data, model):
        """A default predict_fn for TensorFlow. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: TensorFlow model loaded in memory by model_fn

        Returns: a prediction
        """
        return model.predict(input_data)

    def default_output_fn(self, prediction, accept):
        """A default output_fn for TensorFlow. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """

        
        prediction = np.array(prediction).squeeze()
        prediction[prediction > 0.5] = 1
        prediction[prediction <= 0.5] = 0

        # rescaling
        prediction = (prediction * 255).astype(np.uint8)

        return encoder.encode(prediction, accept)


def _json_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a JSON object to a numpy array.
        Args:
            string_like (str): JSON string with -instances key.
            dtype (dtype, optional):  Data type of the resulting array.
                If None, the dtypes will be determined by the contents
                of each column, individually. This argument can only be
                used to 'upcast' the array.  For downcasting, use the
                .astype(t) method.
        Returns:
            (np.array): numpy array
        """
    key = "instances"
    data = json.loads(string_like)[key]
    return np.array(data, dtype=dtype)