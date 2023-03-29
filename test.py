import numpy as np
import cv2
from keras.models import load_model
font = cv2.FONT_HERSHEY_SIMPLEX
import urllib.request
from time import sleep


#############################################
         # PROBABLITY THRESHOLD

##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
model=load_model("model1.h5")

#url='http://192.168.1.14:8080/shot.jpg'
# IMPORT THE TRANNIED MODEL


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getCalssName(classNo):
    if   classNo == 0: return 'Men'
    elif classNo == 1: return 'Women'

    
 

# while True:

    # # READ IMAGE
    # success, imgOrignal = cap.read()

    # PROCESS IMAGE
imgOrignal=cv2.imread("1.jpeg")
img = np.asarray(imgOrignal)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
cv2.imshow("Processed Image", img)
img = img.reshape(1, 32, 32, 1)
cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
# PREDICT IMAGE
predictions = model.predict(img)
classIndex = model.predict_classes(img)
probabilityValue =np.amax(predictions,axis=1)

if probabilityValue > 0.6:
    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue[0]*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            
cv2.imshow("Result", imgOrignal)
cv2.waitKey(0)

# if cv2.waitKey(100) & 0xFF == ord('q'):
cv2.destroyAllWindows()
