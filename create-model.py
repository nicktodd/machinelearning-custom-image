import boto3
import os
#client = boto3.client('sts')
#account = client.get_caller_identity()['Account']

#my_session = boto3.session.Session()

#region = "eu-west-1"

#algorithm_name = 'cifar10-example'

ecr_image =  os.getenv("dockerImage")

print(ecr_image)