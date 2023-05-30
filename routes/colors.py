"""
    Module docstring
"""

from fastapi import APIRouter, Body, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from config.db import Connection
from models.image import Image
from classes.knn import KnnColors

colors = APIRouter()

db_client = Connection(username="admin", password="admin")
db = db_client.connect('fuzzyColor')

@colors.post('/similar_colors', response_model=list[Image] )
async def get_colors(params: dict = Body(...)):
    data = db['objects'].find()
    images = list(data)
    similar_colors = list()
    base_color = params['baseColor']
    delta= int(params['delta'])
    
    for image in images:
        result = KnnColors.compare_colors(color_list=image['predominantColors'], base_color=base_color, delta=delta)
        if result:
            similar_colors.append(image)
    
    return JSONResponse(content=similar_colors, status_code=status.HTTP_200_OK)
