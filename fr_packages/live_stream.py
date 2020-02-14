##################################################
# Dependencies
##################################################
import os
import pickle
import threading

import face_recognition
import cv2
import numpy as np
import importlib

from imutils.video import VideoStream

from fr_packages.find_best_face import find_best_face

##################################################
# Global Variables
##################################################
video_capture = cv2.VideoCapture(0)  # Get a reference to webcam #0

with open('C:\python_api\dataset_faces.dat', 'rb') as f:  # Load Face encodings from dat file
    all_face_encodings = pickle.load(f)

known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []


def live_stream_recognition():

    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = find_best_face(known_face_names, known_face_encodings, face_encoding, 0.4)
                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            if name == "un-identified":
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, top), (right, bottom), (42, 62, 174), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (42, 62, 174), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (38, 73, 28), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (38, 73, 28), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        # cv2.imshow('Video', frame)
        # # Hit 'q' on the keyboard to quit!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    #frame.stop()
