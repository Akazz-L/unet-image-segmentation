# Training container code
## :blue_book: Container code
To build your own training container code, use [AWS SageMaker Training toolkit](https://github.com/aws/sagemaker-training-toolkit). 

The first step is to create a training script [train.py](https://github.com/Akazz-L/unet-image-segmentation/blob/master/train.py) which is going to be the entrypoint of our container.
It contains all the code to run a training job and when deployed with container in AWS the input data and output data will be stored in the following reserved folders.
- **Input data** : /opt/ml/input (the data are copied from S3 when the training job is started on AWS)
- **Output data**: /opt/ml/model (the data is copied to S3 when the training job is done)
- **Code data** : /opt/ml/code

The second step refers to the Docker Image creation wherein all the required packages and libraries that our training environnement needs are installed (python, cuda, Tensorflow etc ...). Make sure to use a base image compatible with *nvidia/cuda* if the training needs GPU support.

To specify train.py as the entrypoint set he **SAGEMAKER_PROGRAM** environment variable to train.py

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

##  :globe_with_meridians: Start a training job

To [start a training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-mkt-algo-train.html#sagemaker-mkt-algo-train-console) job we can either use AWS CLI or AWS Management Console. In both case a training job configuration with the following elements has to be set before starting the process :
- Algorithm source : our own container in ECR (fill in with the registry path)
- Instance : type, count and additional storage
- Maximum runtime 
- Input channel (S3)
- Output path (S3)
- Spot Training (optional)


