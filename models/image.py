"""
    Module docstring
"""

from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId

class PyObjectId(ObjectId):
    """Class docstring"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """Class docstring"""
        if not ObjectId.is_valid(value):
            raise ValueError("Invalid objectid")
        return ObjectId(value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Method docstring"""
        field_schema.update(type="string")

class Image(BaseModel):
    """Class docstring"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias='_id')
    imageId: str =  Field(...)
    imagePath: str = Field(...)
    className: str = Field(...)
    objectName: str = Field(...)
    predominantColors: list[str] = Field(...)
    data: list[dict] = Field(...)

    
    class Config:
        """Class Config docstring"""
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "imageId": "1683221943",
                "imagePath": "https://fuzzy-images-data-gb.s3.amazonaws.com/images/1683221943.jpg",
                "className": "",
                "objectName": "",
                "predominantColors": [
                    "#50443e",
                    "#c4a6a8",
                    "#856e61"
                ],
                "data": [
                    {
                        "Name": "Lipstick",
                        "Confidence": 100.0,
                        "Instances": [
                            {
                                "BoundingBox": {
                                    "Width": 0.14010345935821533,
                                    "Height": 0.39546555280685425,
                                    "Left": 0.48076850175857544,
                                    "Top": 0.06601009517908096
                                },
                                "Confidence": 99.39691162109375
                            }
                        ],
                        "Parents": [
                            {
                                "Name": "Cosmetics"
                            }
                        ],
                        "Aliases": [],
                        "Categories": [
                            {
                                "Name": "Beauty and Personal Care"
                            }
                        ]
                    }                    
                ]
            }
        }
        