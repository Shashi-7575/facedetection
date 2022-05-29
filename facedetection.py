# importing librarys
import cv2
import numpy as np
import face_recognition as face_rec
# function
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

# img declaration
shashi=face_rec.load_image_file('sampe_images/shashi.jpg')
shashi = cv2.cvtColor(shashi, cv2.COLOR_BGR2RGB)
shashi = resize(shashi, 0.50)
shashi_test=face_rec.load_image_file('sampe_images/shashi_test.jpg')
shashi_test = cv2.cvtColor(shashi_test, cv2.COLOR_BGR2RGB)
shashi_test = resize(shashi_test, 0.50)

# finding face location

faceLocation_shashi = face_rec.face_locations(shashi)[0]
encode_shashi = face_rec.face_encodings(shashi)[0]
cv2.rectangle(shashi, (faceLocation_shashi[3], faceLocation_shashi[0]), (faceLocation_shashi[1], faceLocation_shashi[2]), (255, 0, 255), 3)

faceLocation_shashitest = face_rec.face_locations(shashi_test)[0]
encodeshashintest = face_rec.face_encodings(shashi_test)[0]
cv2.rectangle(shashi_test, (faceLocation_shashi[3], faceLocation_shashi[0]), (faceLocation_shashi[1], faceLocation_shashi[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_shashi], encodeshashintest)
print(results)
cv2.putText(shashi_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )


cv2.imshow('main.img',shashi)
cv2.imshow('test.img',shashi_test)

cv2.waitKey(0)
cv2.destroyAllWindows()
