import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file

class Detector:
    def __init__(self) -> None:
        self.modelName = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
        self.cacheDir = './pretrained_models'
        self.colorList = []
        self.classList = ''
        self.colorList = ''
        self.model = ''
    
    def downloadModel(self, modelUrl):
        filename = os.path.basename(modelUrl)
        self.modelName = filename[:filename.index('.')]
        
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=filename, origin=modelUrl, cache_dir=self.cacheDir, cache_subdir="checkpoint", extract=True)

    def readClasses(self, filePath):
        with open(filePath, 'r+') as file:
            self.classList = file.read().splitlines()
            
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classList), 3))
        
    def loadModel(self):
        print('loading model {}...'.format(self.modelName))
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoint", self.modelName, "saved_model"))
        print('Model {} loaded successfully!!!'.format(self.modelName))
    
    def leftSideFence(self, imH, imW, pt):
        # rect = (0, 0, round(imH /2), imW)
        rect = (0, 0, round(imW /2), imH)
        contains = rect[0] <= pt[0] <= rect[0]+rect[2] and rect[1] <= pt[1] <= rect[1]+rect[3]
        return contains
              
    def createBoundingBox(self, image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        
        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
        iou_threshold = threshold, score_threshold = threshold)
        
        # set data about which side an object is.
        imageData = {
                        "data": {
                             "objects" : {
                                "left": [],
                                "right": []
                            }
                        }
                    }
        
        for i in bboxIdx:
            bbox = tuple(bboxs[i].tolist())
            classConfidence = round(100 * classScores[i])
            classIndex = classIndexes[i]
            
            classLabelText = self.classList[classIndex].upper()
            classColor = self.colorList[classIndex]

            displayText = '{}: {}%'.format(classLabelText, classConfidence)

            ymin, xmin, ymax, xmax = bbox
            
            xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
            cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
            # cv2.rectangle(image, (0, 0), (round(imW /2), imW), classColor, 3)
            cv2.line(image, (round(imW /2), 0), (round(imW /2), imH), (0, 0, 255), 1 )
            
            x_middle = round((xmax + xmin) / 2)
            y_middle = round((ymax + ymin) / 2)
            centroid = (x_middle, y_middle)
            
            if self.leftSideFence(imH, imW, centroid):
                imageData["data"]['objects']['left'].append(classLabelText.lower())
            else:
                imageData["data"]['objects']['right'].append(classLabelText.lower())
            
            # print((xmin, ymin), (xmax, ymax), centroid, classLabelText,isLeft)
            
            cv2.circle(image, center=centroid, radius=2, color=classColor, thickness=2)
            #####################################################################
            # lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))

            # cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor,thickness=5)
            # cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor,thickness=5)

            # cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor,thickness=5)
            # cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor,thickness=5)

            # #####################################################################
            # cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor,thickness=5)
            # cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor,thickness=5)

            # cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor,thickness=5)
            # cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor,thickness=5)
        
        jsonObject = json.dumps(imageData, indent=4)
          
        with open('predictions/data.json', 'w+') as data:
            data.write(jsonObject)
            
        return (image, imageData)
    
    def predictImage(self, imagePath, threshold):
        image = cv2.imread(imagePath)
        file = os.path.basename(imagePath)
        filename = file[:file.index('.')]
        imageInfo = self.createBoundingBox(image, threshold)
        
        # TODO: replace file name for a hash for instance.
        cv2.imwrite('predictions/{}.jpg'.format(filename), imageInfo[0])
        # cv2.imshow('Result', bboxImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return imageInfo[1]