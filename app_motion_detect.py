from imutils.video import VideoStream
from flask import Response, request
from flask import Flask
from flask import render_template
import threading
from queue import Queue
import datetime
import imutils
import time
import cv2
import numpy as np
from kthread import KThread

from singlemotiondetector import SingleMotionDetector

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
input_path = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


class InfoCam(object):
    def __init__(self, cam_name):
        self.cap = cv2.VideoCapture(cam_name)
        self.frame_start = 0
        self.total_frame_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_video = int(self.cap.get(cv2.CAP_PROP_FPS))


def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def video_capture(cam, frame_detect_queue):
    frame_count = cam.frame_start
    # frame_step = 2
    # frame_using = 0
    cam.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    while cam.cap.isOpened():
        ret, frame_ori = cam.cap.read()
        # if frame_using != 0 and frame_count % frame_using != 0:
        #     frame_count += 1
        #     continue
        if not ret:
            break

        frame_ori = adjust_gamma(frame_ori, gamma=0.35)
        frame_detect_queue.put([frame_ori, frame_count])
        print("frame_count: ", frame_count)
        frame_count += 1
        # frame_using += frame_step

    cam.cap.release()


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_motion():
    start_time = time.time()
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, input_path
    cv2_show = True
    if cv2_show:
        #  "/storages/data/clover_project/Videos-bk/diemdanh/diem_danh_deo_khau_trang2.mp4"
        input_path = "https://minio.core.greenlabs.ai/clover/motion_detection/motion_clover_2.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20210925%2F%2Fs3%2Faws4_request&X-Amz-Date=20210925T065215Z&X-Amz-Expires=432000&X-Amz-SignedHeaders=host&X-Amz-Signature=3e4b5483e35d7efa8ced818d2c12ec17e4ea2a7e70784678b6182ec3c54aae1f"

    frame_detect_queue = Queue(maxsize=1)
    cam = InfoCam(input_path)
    thread1 = KThread(target=video_capture, args=(cam, frame_detect_queue))
    thread_manager = []
    thread1.daemon = True  # sẽ chặn chương trình chính thoát khi thread còn sống.
    thread1.start()
    thread_manager.append(thread1)

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    # loop over frames from the video stream

    while cam.cap.isOpened():
        frame_ori, frame_count = frame_detect_queue.get()

        frame = imutils.resize(frame_ori, width=1000)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > 32:
            # detect motion in the image
            motion = md.detect(gray, total)
            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()

        if cv2_show:
            cv2.imshow('output', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow('output')
                break

    total_time = time.time() - start_time
    print("FPS video: ", cam.fps_video)
    print("Total time: {}, Total frame: {}, FPS all process : {}".format(total_time, cam.total_frame_video,
                                                                         1 / (total_time / cam.total_frame_video)), )

    for t in thread_manager:
        if t.is_alive():
            t.terminate()
    cv2.destroyAllWindows()


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
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/start_video", methods=['POST'])
def start_video():
    global input_path
    #  https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application
    if request.method == 'POST':
        file = request.form["name"]
        input_path = file
        t = KThread(target=detect_motion)
        t.daemon = True
        t.start()
    # return the rendered template
    return "<p>OK!</p>"


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    app.run(host="0.0.0.0", port="3333", debug=True,
            threaded=True, use_reloader=False)

    # detect_motion()
    # test 2 pc start 2 process khac nhau thi stream se ntn?
