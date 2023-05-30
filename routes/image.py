"""
    Image Module docstring
"""
import os
from fastapi import APIRouter, Body, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from config.db import Connection

image = APIRouter()

db_client = Connection(username="admin", password="admin")
db = db_client.connect('fuzzyColor')

@image.get('/image')
async def get_all_images():
    data = db['objects'].find()
    images = list(data)
    result = jsonable_encoder(images)
    return JSONResponse(content=result, status_code=status.HTTP_200_OK)
