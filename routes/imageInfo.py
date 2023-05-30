"""
    ImageInfo module docstring
"""


import os
import json
import uuid
import numpy
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, File, UploadFile, status
from classes.s3 import S3
from classes.imageTransform import ImageManipulation
from classes.Detector import Detector

UPLOAD_PATH = "uploads"
PREDICTIONS = 'predictions'
BUCKET_S3 = 'https://fuzzy-images-data-gb.s3.amazonaws.com/images'

imageInfo = APIRouter()

threshold = 0.5
classFilePath = "etc/coco.names"
detector = Detector()
detector.readClasses(classFilePath)
detector.loadModel()

@imageInfo.post('/imageInfo')
async def image_info(file: UploadFile = File(...)):
    # modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

    
    try:
        content = file.file.read()
        filename = '{}.jpg'.format(uuid.uuid4())
        file_path = os.path.join(UPLOAD_PATH, filename)
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Reduce image size and quality
        # ImageManipulation.reduce(file_path, quality=40, optimize=True)
    except Exception:
        return {"message": "There was an error uploading the file - {}".format(Exception)}
    finally:
        file.file.close()
        
    

    imageData = detector.predictImage(file_path, threshold)
    imageData["imagePath"] = '{}/{}'.format(BUCKET_S3, filename)
    
    # Persist file on AWS S3
    s3 = S3(bucket_name='fuzzy-images-data-gb')
    s3.write(os.path.join(PREDICTIONS, filename))
    
    return JSONResponse(content=imageData, status_code=status.HTTP_200_OK)
        
    