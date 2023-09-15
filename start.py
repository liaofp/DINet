import os
from config.config import DINetInferenceOptions
from inference2 import inference
import cv2
import subprocess as sp
from typing import Generator
from argparse import Namespace

DST = "/home/developer/test.mp4"

def read_video(capture: cv2.VideoCapture)-> Generator:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yield frame

def push_stream(config: Namespace)-> None:
    res_video_path = os.path.join(config.res_video_dir, os.path.basename(config.source_video_path)[:-4] + '_synthetic_face.mp4')
    if not os.path.exists(res_video_path):
        live_stream(config)
    else:
        file_stream(res_video_path)

def file_stream(video_path: str):
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # h = int(capture.get(cv2.CAP_PROP_FOURCC))
    # codec = chr(h&0xff) + chr((h>>8)&0xff) + chr((h>>16)&0xff) + chr((h>>24)&0xff)
    print(f"fps={fps}, width={width}, height={height}")
    command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(width, height),
           '-r', str(fps),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           '-g', '5',
           DST]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    for frame in read_video(capture):
        # pipe.communicate(input=frame.tobytes())
        pipe.stdin.write(frame.tobytes())
    capture.release()
    pipe.stdin.close()

def live_stream(config: Namespace):
    command = ["ffmpeg",
               "-i", "-",
               "-i", "{}".format(config.driving_audio_path),
               "-c:v", "copy",
               "-c:a", "aac",
               "-strict", "experimental",
               "-map", "0:v:0",
               "-map", " 1:a:0",
               DST]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    frames = inference(config)
    for frame in frames:
        pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()


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
    




