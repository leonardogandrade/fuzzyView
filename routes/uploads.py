"""
    Upload files module
"""


import os
import uuid
import json
import numpy
import time
# from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, File, UploadFile, status
from classes.rekognition import ImageLabelDetection
from models.image import Image
from config.db import Connection
from classes.s3 import S3
from classes.imageTransform import ImageManipulation
from classes.knn import KnnColors

UPLOAD_PATH = "uploads"
BUCKET_S3 = 'https://fuzzy-images-data-gb.s3.amazonaws.com/images'

upload = APIRouter()

db_client = Connection(username="admin", password="admin")
db = db_client.connect('fuzzyColor')

def create_payload(file_uuid, filename, image_data, predominant_colors):
        labels_data = dict()
    
        # labels_data['imageId'] = bson.Binary.from_uuid(file_uuid)
        labels_data['imageId'] = file_uuid
        labels_data['imagePath'] = os.path.join(BUCKET_S3, filename)
        labels_data['predominantColors'] = jsonable_encoder(predominant_colors)
        
        if len(image_data) > 0:
            labels_data['className'] = image_data[0]['Categories'][0]['Name'] or ''
            labels_data['objectName'] = image_data[0]['Name'] or ''
            labels_data['data'] = image_data
        else:
            labels_data['className'] = ''
            labels_data['objectName'] = ''
            labels_data['data'] = image_data
    
        payload = Image(**labels_data)
        
        return jsonable_encoder(payload)

@upload.post('/upload', response_model=Image)
async def upload_file(classFilter: str, file: UploadFile = File(...)):
    
    file_uuid = round(time.time())
    
    try:
        
        content = file.file.read()
        filename = f'{file_uuid}.jpg'
        file_path = os.path.join(UPLOAD_PATH, filename)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Reduce image size and quality
        ImageManipulation.reduce(file_path, quality=40, optimize=True)
        
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    # Get predominant collors by KNN
    
    skip_colors_list = {
        'grey': [191, 191, 191],
        'white': [255, 255, 255]
    }

    clusters = 3
    knn = KnnColors(file_path, clusters)
    predominant_colors = knn.get_colors()
    
    # result_skip_colors = knn.skip_colors(file_path, skip_colors=skip_colors_list, color_fuzzy_distance= 60)
    # print(result_skip_colors)

    #Object name
    object_name = "Lipstick"
    payload = {}
    image_data = []
           
    if classFilter == object_name:
        # Image labels detection
        detector = ImageLabelDetection()
        image_labels = detector.detect_labels(file_path)

        image_data = [data for data in image_labels['data'] if data['Name'] == object_name]


    payload = create_payload(file_uuid, filename, image_data, predominant_colors)
    
    # Persist data on mongo cloud
    db['objects'].insert_one(payload)
    
    # Persist file on AWS S3
    print(file_path)
    s3 = S3(bucket_name='fuzzy-images-data-gb')
    s3.write(file_path)
    
    return JSONResponse(content=payload, status_code=status.HTTP_200_OK)
