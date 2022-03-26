import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('src\cascade\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('src\cascade\data\haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


labels = {}
with open("label.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
image_name_id = 180


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # indicate faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        id_, conf = recognizer.predict(roi_gray)
        if conf >= 4:#5 and conf <= 85:
            font = cv2.FONT_HERSHEY_COMPLEX
            name =  labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            # img_item = 'src\img\ '[0:-1] + name + "\ "[0:-1] + str(image_name_id) +".png"
            # image_name_id = image_name_id + 1
            # print(img_item, image_name_id)
            # cv2.imwrite(img_item, frame)
            # if image_name_id == 600:
            #     image_name_id = 10

        #drawing rectangle
        color = (255, 0, 0) #color BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = h + y
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)



    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()