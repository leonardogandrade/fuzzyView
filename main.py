"""
    Main module
"""


from fastapi import FastAPI
from routes.image import image
from routes.user import user
from routes.colors import colors
from routes.uploads import upload
from routes.imageInfo import imageInfo
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = {
    'http://localhost:3000',
    'http://localhost',
    'http://domain.local'
}


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload)
app.include_router(colors)
app.include_router(imageInfo)
