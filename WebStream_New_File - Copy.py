
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
from fr_packages.load_data import load_trained_data
import time
import face_recognition
from imutils.video import VideoStream
from fr_packages.find_best_face import find_best_face
from fr_packages.live_stream import video_capture
import cv2
import numpy as np
import pickle
import os

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


# def detect_motion(frameCount):
#     # grab global references to the video stream, output frame, and
#     # lock variables
#     global vs, outputFrame, lock
#
#     # initialize the motion detector and the total number of frames
#     # read thus far
#     md = SingleMotionDetector(accumWeight=0.1)
#     total = 0
#
#     # loop over frames from the video stream
#     while True:
#         # read the next frame from the video stream, resize it,
#         # convert the frame to grayscale, and blur it
#         frame = vs.read()
#         frame = imutils.resize(frame, width=400)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
#         # grab the current timestamp and draw it on the frame
#         timestamp = datetime.datetime.now()
#         cv2.putText(frame, timestamp.strftime(
#             "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#
#         # if the total number of frames has reached a sufficient
#         # number to construct a reasonable background model, then
#         # continue to process the frame
#         if total > frameCount:
#             # detect motion in the image
#             motion = md.detect(gray)
#
#             # cehck to see if motion was found in the frame
#             if motion is not None:
#                 # unpack the tuple and draw the box surrounding the
#                 # "motion area" on the output frame
#                 (thresh, (minX, minY, maxX, maxY)) = motion
#                 cv2.rectangle(frame, (minX, minY), (maxX, maxY),
#                               (0, 0, 255), 2)
#
#         # update the background model and increment the total number
#         # of frames read thus far
#         md.update(gray)
#         total += 1
#
#         # acquire the lock, set the output frame, and release the
#         # lock
#         with lock:
#             outputFrame = frame.copy()

def live_stream_recognition():
    global vs, outputFrame, lock
    trained_face_encodings = load_trained_data()
    print(trained_face_encodings)
    known_face_names = list(trained_face_encodings.keys())
    known_face_encodings = np.array(list(trained_face_encodings.values()))


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []

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

        with lock:
            outputFrame = frame.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    t = threading.Thread(target=live_stream_recognition)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()