import os
from config.config import DINetInferenceOptions
from inference2 import inference
import cv2
import subprocess as sp
from typing import Generator
from argparse import Namespace

DST = "rtmp://192.168.1.119/live/ai.flv"

def read_video(capture: cv2.VideoCapture)-> Generator:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yield frame

def push_stream(config: Namespace)-> None:
    res_video_path = os.path.join(config.res_video_dir, os.path.basename(config.source_video_path)[:-4] + "_facial_dubbing.mp4")
    # res_video_path = r""

    if not os.path.exists(res_video_path):
        live_stream(config)
    else:
        file_stream(res_video_path, config.driving_audio_path)

def file_stream(video_path: str, audio_path: str):
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = ['ffmpeg',
        '-y',
        '-re', # '-re' is requiered when streaming in "real-time"
        '-f', 'rawvideo',
        #'-thread_queue_size', '1024',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', "{}x{}".format(width, height),
        '-r', str(fps),
        '-i', '-',
        "-i", audio_path,
        # '-vn', '-i', audio_path,  # Get the audio stream without using OpenCV
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-c:a', 'aac',  # Select audio codec
        '-bufsize', '64M',  # Buffering is probably required
        '-f', 'flv', 
        DST]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    while (capture.isOpened()):
        ret, frame = capture.read()
        if not ret:
            print("End of input file")
            break
        # write to pipe
        pipe.stdin.write(frame.tobytes())
    capture.release()
    pipe.stdin.close()

def live_stream(config: Namespace):
    capture = cv2.VideoCapture(config.driving_audio_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    command = ['ffmpeg',
        '-y',
        '-re', # '-re' is requiered when streaming in "real-time"
        '-f', 'rawvideo',
        #'-thread_queue_size', '1024',  # May help https://stackoverflow.com/questions/61723571/correct-usage-of-thread-queue-size-in-ffmpeg
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
         '-s', "{}x{}".format(width, height),
         '-r', str(25),
        '-i', '-',
        "-i", config.driving_audio_path,
        # '-vn', '-i', audio_path,  # Get the audio stream without using OpenCV
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-c:a', 'aac',  # Select audio codec
        '-bufsize', '64M',  # Buffering is probably required
        '-f', 'flv', 
        DST]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    frames = inference(config)
    for frame in frames:
        pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()
    pipe.kill()


if __name__ == "__main__":
    config = DINetInferenceOptions().parse_args()
    if not os.path.exists(config.source_video_path):
        raise ('wrong video path : {}'.format(config.source_video_path))
    if not os.path.exists(config.deepspeech_model_path):
        raise ('pls download pretrained model of deepspeech')
    if not os.path.exists(config.driving_audio_path):
        raise ('pls download pretrained model of deepspeech')
    if not os.path.exists(config.source_openface_landmark_path):
        raise ('wrong facial landmark path :{}'.format(config.source_openface_landmark_path))
    push_stream(config)
    




