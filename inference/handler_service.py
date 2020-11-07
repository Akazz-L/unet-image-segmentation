from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer
from inference_handler import TensorFlowInferenceHandler
import os

class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/multi-model-server/blob/master/docs/custom_service.md
    """
    def __init__(self):
        transformer = Transformer(default_inference_handler=TensorFlowInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)

    def initialize(self, context):

        """ 
        Calls the Transformer method that validates the user module against
        the SageMaker inference contract.
        """

        #properties = context.system_properties
        #model_dir = properties.get("model_dir")
        model_dir = '/opt/ml/model' # Model directory using AWS Sagemaker customized inference

        # add model_dir/code to python path
        code_dir_path = "{}:".format(model_dir + "/code")
        if PYTHON_PATH_ENV in os.environ:
            os.environ[PYTHON_PATH_ENV] = code_dir_path + os.environ[PYTHON_PATH_ENV]
        else:
            os.environ[PYTHON_PATH_ENV] = code_dir_path

        self._service.validate_and_initialize(model_dir=model_dir)

