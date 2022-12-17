import numpy as np
import face_recognition as fr
import cv2
from engine import get_rostos

rostos_conhecidos, nomes_dos_rostos = get_rostos()

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    #Forloop para cada uma das faces encontradas na webcam
    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        results = fr.compare_faces(rostos_conhecidos, face_encodings)

        face_distances = fr.face_distance(rostos_conhecidos, face_encodings)

        melhor_id = np.argmin(face_distances)
        if(results[melhor_id]):
            name = nomes_dos_rostos[melhor_id]
        else:
            name = "Desconhecido"


        #Quadrado ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #Embaixo do rosto
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX

        #Texto
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('webcam_facerecognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()