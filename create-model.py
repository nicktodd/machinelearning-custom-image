
import boto3
import sagemaker
from sagemaker import get_execution_role

client = boto3.client('sts')
account = client.get_caller_identity()['Account']


role = "arn:aws:iam::963778699255:role/service-role/AmazonSageMaker-ExecutionRole-20190723T151113" #get_execution_role()

region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
bucket = sagemaker_session.default_bucket()
prefix = 'custom-ml-image'

print(region)
print(role)
print(bucket)

# for now we will assume the tag is always the latest
ecr_image =  account + ".dkr.ecr.eu-west-1.amazonaws.com/custommlimage:latest"

session = sagemaker.Session()

print(session.upload_data('dummy.csv', bucket, prefix + '/train'))
print(session.upload_data('dummy.csv', bucket, prefix + '/val'))


est = sagemaker.estimator.Estimator(ecr_image,
                                    role, 
                                    train_instance_count=1, 
                                    #train_instance_type='local', # use local mode
                                    train_instance_type='ml.m5.xlarge', # can't use local when running in CodeCommit
                                    base_job_name=prefix)


est.set_hyperparameters(hp1='value1',
                        hp2=300,
                        hp3=0.001)

train_config = sagemaker.session.s3_input('s3://{0}/{1}/train/'.format(bucket, prefix), content_type='text/csv')
val_config = sagemaker.session.s3_input('s3://{0}/{1}/val/'.format(bucket, prefix), content_type='text/csv')

est.fit({'train': train_config, 'validation': val_config })

est.deploy(1, "ml.m5.xlarge")


# now let's create the cloudformation template parameters file ready for the CodeDeploy step in the pipeline

#model_data_url = 's3://{}/{}/output'.format(bucket, project_name) + "/" + training_job_name + "/output/model.tar.gz"

model_data_s3_path = est.model_data
print("S3 path to model is " + model_data_s3_path)


parameter_file_data = {
    
        "Parameters" : {
            "ModelName" : "CustomMLModel",
            "ModelDataUrl" : model_data_s3_path,
            "TrainingImage": ecr_image,
            "InstanceType" : "ml.t2.xlarge",
            "InstanceCount": str(1),
            "RoleArn": role
        }
    
}

print("using this parameter file")
print(parameter_file_data)



print("Now waiting for the step function pipeline to complete. This can take up to 5 minutes")
# can be commented out to avoid unnecessary costs of keeping build container alive



import json
with open('cloudformation_parameters.json', 'w') as outfile:    
        json.dump(parameter_file_data, outfile)

