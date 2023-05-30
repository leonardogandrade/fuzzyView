"""
    Image manipulation module
"""


from PIL import ImageOps, Image

class ImageManipulation:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def crop_image(image, bbox):
        original_img = Image.open(image)
        width, height = original_img.size

        left = width * bbox['Left']
        top = height * bbox['Top']
        right = (width *  bbox['Width']) + left
        bottom = (height * bbox['Height']) + top
        
        cropped_img = original_img.crop((left, top, right, bottom))
        cropped_img.save('croped.jpg')
    
    @staticmethod
    def reduce(image_path, quality, optimize=True):
        image = Image.open(image_path)
        new_image = image.resize((400, 600))
        new_image.save(image_path, quality= quality, optimize=optimize)


# boundingBox1 = {
#     "Width": 0.13893763720989227,
#     "Height": 0.396090030670166,
#     "Left": 0.478204607963562,
#     "Top": 0.06691939383745193
# }

# boundingBox2 = {
#     "Width": 0.2926446497440338,
#     "Height": 0.5362605452537537,
#     "Left": 0.32682475447654724,
#     "Top": 0.17931142449378967
# }

# ImageManipulation.crop_image('classes/batom_2.jpeg', bbox=boundingBox2)