import boto3
import io
import yaml

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)



s3 = boto3.resource('s3',
                    aws_access_key_id=cfg["aws"]["aws_access_key_id"],
                    aws_secret_access_key= cfg["aws"]["aws_secret_access_key"])

for bucket in s3.buckets.all():
    print(bucket.name)

def upload_to_folder(folder_name, filename, bucketname, file):

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    s3 = boto3.client('s3',
                    aws_access_key_id=cfg["aws"]["aws_access_key_id"],
                    aws_secret_access_key= cfg["aws"]["aws_secret_access_key"])
    
    file_name = folder_name + '/' + filename

    print(type(file))
    try:
        s3.upload_fileobj(file, bucketname, file_name)
        #s3.upload_file(file_name, bucketname, file)
    except Exception as e:
        print(str(e))
        return e
    

def download_file_bytes(folder_name):

    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    s3 = boto3.client('s3',
                    aws_access_key_id=cfg["aws"]["aws_access_key_id"],
                    aws_secret_access_key= cfg["aws"]["aws_secret_access_key"])

    try:
        data = io.BytesIO()
        s3.download_fileobj(Bucket='clotoure', Key=folder_name, Fileobj=data)
        data.seek(0)
        return data
    except Exception as e:
        print(str(e))
        return e
    


