"""
    Module docstring
"""

from fastapi import APIRouter, Body, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from models.user import User
from config.db import Connection

user = APIRouter()

db_client = Connection("admin", "admin")
db = db_client.connect('college')

@user.post('/user')
def create_user(objectClassName: str, data: User = Body(...)):
    """Method docstring"""
    
    # _user = jsonable_encoder(data)
    # new_user = db['user'].insert_one(_user)
    # response = db['user'].find_one({"_id": new_user.inserted_id})
    return JSONResponse(content='response', status_code=status.HTTP_200_OK)


@user.get('/user', response_model=list[User])
async def get_all_users():
    data = db['user'].find()
    users = list(data)
    return JSONResponse(content=users, status_code=status.HTTP_200_OK)
