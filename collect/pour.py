from pynput.keyboard import Listener, Key
import threading
import time
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import signal
from gelsight import gsdevice
import pyaudio
import wave
# import torch

global_time = time.time()
press = False
press_type = 0
arm = None
init = False
f_img = None
f_tactile = None
f_audio = None
frames = []

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

RECORD_SECONDS = 3

def on_press(key):
    global press, press_type, init
    if init:
        if key == Key.up:
            press = True
            press_type = 3
        elif key == Key.down:
            press = True
            press_type = 4
        elif key == Key.left:
            press = True
            press_type = 2
        elif key == Key.right:
            press = True
            press_type = 1
        elif key == Key.enter:
            press = True
            press_type = 7
        elif str(key) == '\'w\'':
            press = True
            press_type = 5
        elif str(key) == '\'s\'':
            press = True
            press_type = 6
            # global_time = time.time()
    
def on_release(key):
    global press, press_type, init
    if init:
        if key == Key.up:
            press = False
            #press_type = 0
        elif key == Key.down:
            press = False
            #press_type = 0
        elif key == Key.left:
            press = False
            #press_type = 0
        elif key == Key.right:
            press = False
            #press_type = 0
        elif str(key) == '\'w\'':
            press = False
        elif str(key) == '\'s\'':
            press = False

def func_audio():
    global f_audio
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)
    
    f_audio = open('data_pour/03audio.txt', 'w')
    f_audio.write(str(time.time())+'\n')
    while True:
        data_pour = stream.read(CHUNK)
        frames.append(data_pour)
    
    

def func_listen():
    with open('data_pour/00record.txt', 'w') as f:
        global press, press_type, init, f_img, f_tactile, f_audio
        while True:
            if init:
                _, position = arm.get_position()
                if press:
                    if press_type == 1:
                        print('forward')
                        arm.set_position(x=0.75, relative=True, speed=6, wait=False)
                    elif press_type == 2:
                        print('backward')
                        arm.set_position(x=-0.75, relative=True, speed=6, wait=False)
                    elif press_type == 3:
                        print('back')
                        arm.set_position(pitch=0.2, relative=True, speed=6, wait=False)
                    elif press_type == 4:
                        print('pour')
                        arm.set_position(pitch=-0.2, relative=True, speed=6, wait=False)
                    # elif press_type == 5:
                    #     print('up')
                    #     arm.set_position(z=1.0, relative=True, speed=10, wait=False)
                    # elif press_type == 6:
                    #     print('down')
                    #     arm.set_position(z=-1.0, relative=True, speed=10, wait=False)
                    elif press_type == 7:
                        wf = wave.open('data_pour/audio.wav', 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        f_audio.write(str(time.time())+'\n')
                        wf.writeframes(b''.join(frames))
                        wf.close()
                        f.close()
                        f_img.close()
                        f_tactile.close()
                        f_audio.close()
                        os.kill(os.getpid(), signal.SIGINT)
                    f.write(str(press_type)+','+str(time.time())+','+str(position)+'\n')
                else:
                    arm.set_position(pitch=0.0, relative=True, speed=6, wait=False)
                    print('wait')
                    f.write('0,'+str(time.time())+','+str(position)+'\n')
                time.sleep(0.1995)

def func_img():
    global f_img,init,f_tactile
    i = 0
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pf = pipeline.start(config)
    for t in range(20):
        frame = pipeline.wait_for_frames()
    with open('data_pour/01img.txt', 'w') as f:
        f_img = f
        while True:
            frame = pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite('data_pour/rgb/'+str(i)+'.png', color_image)
            cv2.imwrite('test2.png', color_image)
            f.write(str(i)+','+str(time.time())+'\n')
            i += 1
            if i==120:
                print('start!')
                f.write('start\n')
                f_tactile.write('start\n')
                init=True

def func_tactile():
    global f_tactile
    i = 0
    finger = gsdevice.Finger.MINI
    cam_id = gsdevice.get_camera_id("GelSight Mini")
    dev = gsdevice.Camera(finger, cam_id)
    dev.connect()
    
    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])
    
    with open('data_pour/02tactile.txt', 'w') as f:
        f_tactile = f
        while dev.while_condition:
            f1 = dev.get_image(roi)
            cv2.imwrite('data_pour/tactile/'+str(i)+'.png', f1)
            cv2.imwrite('test3.png', f1)
            f.write(str(i)+','+str(time.time())+'\n')
            i += 1
    
def func_key():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        
if __name__ == '__main__':
    if not os.path.exists('data_pour'):
        os.mkdir('data_pour')
    if not os.path.exists('data_pour/rgb'):
        os.mkdir('data_pour/rgb')  
    if not os.path.exists('data_pour/tactile'):
        os.mkdir('data_pour/tactile')   
    ip = "192.168.1.224"
    arm = XArmAPI(ip, is_radian=False)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_tcp_load(1.025,[0,0,73])
    
    # print(arm.get_position())
    # exit(0)
    
    tt = threading.Thread(target=func_key)
    tt.start()
    # arm.set_position(x=10, relative=True, speed=30, wait=True)
    tt2 = threading.Thread(target=func_listen)
    tt2.start()
    
    tta = threading.Thread(target=func_audio)
    tta.start()
    
    tti = threading.Thread(target=func_img)
    tti.start()
    
    ttt = threading.Thread(target=func_tactile)
    ttt.start()
    
    