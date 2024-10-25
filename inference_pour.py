import torch
from models.model_pour import ConcatModel, MULSA, MSBot
import time
import threading
import pyaudio
import wave
import os
import pyrealsense2 as rs
import cv2
import numpy as np
from gelsight import gsdevice
from PIL import Image
import torchvision.transforms as T
import torchaudio
from xarm.wrapper import XArmAPI
import signal
from config import parse_args

CHUNK = 441
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

sr = 16000
mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64,
            center=False,
        )

frames = []
last_img = 0
now_img = 0
img_time = []
last_tact = 0
now_tact = 0
now_audio = 0
tact_time = []

transform= T.Compose(
                [
                    T.Resize((105, 140)),
                    T.CenterCrop((96, 128)),
                    T.ToTensor(),
                    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                ]
            )

def func_audio():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK)

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

def func_img():
    global now_img
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pf = pipeline.start(config)
    for t in range(20):
        frame = pipeline.wait_for_frames()
    
    while True:
        frame = pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite('temp_data/rgb/'+str(now_img)+'.png', color_image)
        img_time.append(time.time())
        # cv2.imwrite('test2.png', color_image)
        now_img += 1
        
def func_tactile():

    global now_tact
    finger = gsdevice.Finger.MINI
    cam_id = gsdevice.get_camera_id("GelSight Mini")
    dev = gsdevice.Camera(finger, cam_id)
    dev.connect()
    
    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])
    while dev.while_condition:
        f1 = dev.get_image(roi)
        cv2.imwrite('temp_data/tactile/'+str(now_tact)+'.png', f1)
        tact_time.append(time.time())
        now_tact += 1


def get_data():
    global now_img, now_tact, last_img, last_tact, now_audio
    
    wf = wave.open('temp_data/audio/'+str(now_audio)+'.wav', 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames[int(-RATE / CHUNK *3):]))
    wf.close()
    
    EPS = 1e-8
    audio,_ = torchaudio.load('temp_data/audio/'+str(now_audio)+'.wav')
    audio = torchaudio.functional.resample(audio, 44100, 16000)
    audio = audio.view(1, audio.size()[0], 6, -1)
    audio = mel(audio.float())
    audio = torch.log(audio+ EPS)
    
    now_audio += 1
    
    while img_time[last_img] + 3.0 < img_time[now_img-1]:
        last_img += 1
    
    while tact_time[last_tact] + 3.0 < tact_time[now_tact-1]:
        last_tact += 1
    
    clip_img = (now_img - last_img)//5
    clip_tact = (now_tact - last_tact)//5

    
    img_arr = []
    tact_arr = []

    for i in range(6):

        img = Image.open('temp_data/rgb/'+str(last_img + i * clip_img - (i==5)*1)+'.png').convert('RGB')
        tact = Image.open('temp_data/tactile/'+str(last_tact + i * clip_tact - (i==5)*1)+'.png').convert('RGB')

        img_arr.append(transform(img).unsqueeze(0).unsqueeze(0))
        tact_arr.append(transform(tact).unsqueeze(0).unsqueeze(0))

        
    image = torch.cat(img_arr, dim=1).permute(0, 2, 1, 3, 4)
    tactile = torch.cat(tact_arr, dim=1).permute(0, 2, 1, 3, 4)
    
    return audio.cuda(), image.cuda(), tactile.cuda()
    
def move_arm(arm, action):
    if action == -1: # not used
        print('right')
        arm.set_position(x=0.75, relative=True, speed=6, wait=False)
    elif action == 0:
        print('forward')
        arm.set_position(x=-0.75, relative=True, speed=6, wait=False)
    elif action == 3:
        print('back')
        arm.set_position(pitch=0.2, relative=True, speed=6, wait=False)
    elif action == 1:
        print('pour')
        arm.set_position(pitch=-0.2, relative=True, speed=6, wait=False)

def main(args):
    ip = "192.168.1.224"
    arm = XArmAPI(ip, is_radian=False)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_tcp_load(1.025,[0,0,73])
    
    seq_len = args.seq_len
    weight_prompt = args.inference_weight
    actions = torch.ones(1, seq_len, 4).cuda()
    
    if args.model == 'Concat':
        model = ConcatModel().cuda()
    elif args.model == 'MULSA':
        model = MULSA().cuda()
    elif args.model == 'MSBot':
        model = MSBot().cuda()
    else:
        print('Model not found')
        exit(0) 
    
    loaded_dict = torch.load(args.model_dir)['model']
    model.load_state_dict(loaded_dict, strict = True)

    img = torch.ones(1,3,6,96,128).cuda()
    tact = torch.ones(1,3,6,96,128).cuda()
    audio = torch.ones(1, 1, 6, 64, 48).cuda()

    weight_prompt = torch.tensor(weight_prompt).unsqueeze(0).unsqueeze(0).cuda()

    a = model(audio, img, tact, actions, weight_prompt.int())
    softmax = torch.nn.Softmax()

    if not os.path.exists('temp_data'):
        os.mkdir('temp_data')
    if not os.path.exists('temp_data/rgb'):
        os.mkdir('temp_data/rgb')   
    if not os.path.exists('temp_data/audio'):
        os.mkdir('temp_data/audio')  
    if not os.path.exists('temp_data/tactile'):
        os.mkdir('temp_data/tactile')   
    
    tta = threading.Thread(target=func_audio)
    tta.start()
    
    tti = threading.Thread(target=func_img)
    tti.start()
    
    ttt = threading.Thread(target=func_tactile)
    ttt.start()
    
    his_seq = [torch.zeros(1,4)] * seq_len
    
    time.sleep(5)
    
    back = 0
    
    last_t = time.time()
    while True:
        now_t = time.time()
        if now_t - last_t >= 0.2:
            last_t = now_t
            audio, img, tact = get_data()
            history = torch.cat(his_seq[-seq_len:]).unsqueeze(0).cuda()

            with torch.no_grad():
                model.eval()
                out, score = model(audio.float(), img.float(), tact.float(), history.float(), weight_prompt.int())

            action = int(torch.argmax(out))
            onehot_action = torch.zeros(1,4)
            onehot_action[0][action] = 1.0
            his_seq.append(onehot_action)

            move_arm(arm, action)
            print(action, time.time())

            if(action == 3):
                back += 1

            if back >= 40:
                # task completed
                os.kill(os.getpid(), signal.SIGINT)

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)