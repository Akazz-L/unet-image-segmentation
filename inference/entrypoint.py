from sagemaker_inference import model_server
from . import handler_service


HANDLER_SERVICE = handler_service.__file__
model_server.start_model_server(handler_service=HANDLER_SERVICE)