import boto3
import os
import sagemaker
from utils.utils_cifar import get_train_data_loader, get_test_data_loader, imshow, classes

client = boto3.client('sts')
account = client.get_caller_identity()['Account']
region = "eu-west-1"
algorithm_name = 'cifar10-example'
# for now we will assume the tag is always the latest
ecr_image =  account + ".dkr.ecr.eu-west-1.amazonaws.com/cifar10example:latest"

session = sagemaker.Session()

trainloader = get_train_data_loader('/tmp/pytorch-example/cifar-10-data')
testloader = get_test_data_loader('/tmp/pytorch-example/cifar-10-data')

prefix = 'DEMO-pytorch-cifar10'
WORK_DIRECTORY = '/tmp/pytorch-example/cifar-10-data'
data_location = session.upload_data(WORK_DIRECTORY, key_prefix=prefix)