import os
import json
import boto3
import time

class ImageLabelDetection:
    
    def __init__(self) -> None:
        self.client = boto3.client('rekognition')

    def detect_labels(self, photo):
        
        with open(photo, 'rb') as image:
            response = self.client.detect_labels(Image={'Bytes': image.read()})
            
            data = {
                'source': photo,
                'data': response.get('Labels')
            }
            
        return data
