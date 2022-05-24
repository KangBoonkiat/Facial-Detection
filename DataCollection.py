# Import Libraries
import time 
import uuid
import cv2

# Gathering Imaga Data using webcam


#Set the number of image to take
total_image = 5

#Open the webcam and take the set number of image.
#Save those images with unqiue ID to the same folder as this .py file
camera = cv2.VideoCapture(0)
i = 0
while i < total_image:
    print ('total image {}'.format(i))
    return_value, image = camera.read()
    cv2.imwrite(f'{str(uuid.uuid1())}.jpg', image)
    cv2.imshow('frame',image)
    time.sleep(0.5)
    i += 1
del(camera)

cv2.destroyAllWindows()