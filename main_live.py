import argparse
import threading
import time

from imutils.video import VideoStream

from fr_packages.live_stream import live_stream_recognition
# from imutils.video import VideoStream
# from flask import Response
# from flask import Flask
# from flask import render_template
# import threading
# import argparse
# import datetime
# import imutils
# import time
from flask import render_template, app, Response, Request
import cv2

from flask import Flask


app = Flask(__name__)
@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def generate():
        # yield the output frame in the byte format
        while True:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			    bytearray(live_stream_recognition()) + b'\r\n')
        video_capture.release()
        cv2.destroyAllWindows()

@app.route("/video_feed")
def video_feed():
    return Response(generate())


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,help="# of frames used to construct the background model")

    args = vars(ap.parse_args())
    t = threading.Thread(target=live_stream_recognition, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, use_reloader=False)
