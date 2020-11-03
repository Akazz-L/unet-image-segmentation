from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors
import os
import textwrap
from model import *

MODEL_FILENAME = "unet_membrane.hdf5"




class TensorFlowInferenceHandler(default_inference_handler.DefaultInferenceHandler):

    def model_fn(self, model_dir):
        model_path = os.path.join(model_dir, MODEL_FILENAME)    
        if not os.path.exists(model_path):
            raise FileNotFoundError("Failed to load model with default model_fn: missing file {}.".format(DEFAULT_MODEL_FILENAME))
        model = unet_model(pretrained_weights = model_path)
        return model


    def input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into Numpy Array (Batch_size, width, height, channel)for tensorflow model
        """
        input_data = decoder.decode(input_data, content_type)
        input_data = input_data / 255 
        return decoder.decode(input_data, content_type)

    def predict_fn(self, input_data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn

        Returns: a prediction
        """
        return model.predict(input_data)

    def output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        return encoder.encode(prediction, accept)