# Inference container code
## :blue_book: Container code
To build your own inference container code, use [AWS SageMaker Inference toolkit](https://github.com/aws/sagemaker-inference-toolkit). This package is based on the multi-model-server (MMS) and allows single model and multiple models serving and provides a lot of useful tools like serializing encoder, decoder, content-types, endpoint requests handler ...

The tool eases model deployment and only a few files are need to get our inference container ready.

- [entrypoint.py](https://github.com/Akazz-L/unet-image-segmentation/blob/master/inference/entrypoint.py) starts the model-server provided by the inference toolkit which is going to handle the inference requests through our handler service.  

- [handler_service.py](https://github.com/Akazz-L/unet-image-segmentation/blob/master/inference/handler_service.py) contains two default methods :
    - 'handle' method  which retrieves inference requests and redirect them to the inference handler after validation if needed
    - 'initialize' method which configures our machine learning model when the multi-model-server is started

- [inference_handler.py](https://github.com/Akazz-L/unet-image-segmentation/blob/master/inference/inference_handler.py) contains all the model logic including the model architecture, the input decoder, pre-processing, the model prediction and the output processing. All these methods are sequentially called by the sagemaker-inference [transformer](https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/transformer.py) object which is initialized with this inference handler.

*Note : It is not necessary to separate the logic into three files.*

Finally a [Dockerfile](https://github.com/Akazz-L/unet-image-segmentation/blob/master/inference/Dockerfile) assembles the image by configuring the environment, installing the python packages (tensorflow, keras,sagemaker-inference multi-model-server ...) and copying the above files in the container and run the entrypoint.

## :whale: AWS ECR (Docker registry)
As we will use AWS SageMaker to serve our model, we can also use [AWS ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html) as our docker registry that will store our docker image.
1. Login to AWS ECR from CLI by replacing AWS_ACCOUNT_ID and REGION with your own credentials
```
aws ecr get-login-password | docker login --username AWS --password-stdin AWS_ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
```
2. Create a docker repository using the AWS ECR service (AWS Management Console) or with AWS-CLI and provide the repository name.
```
aws ecr create-repository --repository-name [REPO_NAME]
```
3. Build and push

```
docker build -t [REPO_URI] .
docker push [REPO_URI]
```



##  :globe_with_meridians: Container Deployment

### Model

On AWS SageMaker Service, create the model  by providing :
- model artifacts location (S3)
- inference image location (ECR)

### Endpoint Configuration

Then create the endpoint configuration including :
- the models that are going to be served
- the HW resources 

### Endpoint
Finally create the endpoin by selecting the previous endpoint configuration which will automatically start an instance and deploy the container