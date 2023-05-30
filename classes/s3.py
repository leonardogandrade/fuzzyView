"""
    S3 Bucket module
"""


import os
import boto3
import logging


class S3:
    def __init__(self, bucket_name, region_name = 'us-east-1') -> None:
        self.bucket_name = bucket_name
        self.region_name = 'us-east-1'
    
    def write(self, file):
        try:
            filename = os.path.basename(file)
            client = boto3.client('s3', self.region_name)
            client.upload_file(file, self.bucket_name , 'images/{}'.format(filename))
        except Exception:
            logging.error('Error writing file on S3 Bucket : {}'.format(Exception))


# s3 = S3(bucket_name='fuzzy-images-data-gb')
# s3.write(os.path.join('classes', 'batom_1.jpg'))