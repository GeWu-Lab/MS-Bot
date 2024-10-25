import torch
from models.model_peg import ConcatModel, MULSA, MSBot
import time
import threading
import os
import pyrealsense2 as rs
import cv2
import numpy as np
from gelsight import gsdevice
from PIL import Image
import torchvision.transforms as T
from xarm.wrapper import XArmAPI
import signal
from config import parse_args

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


def func_img():
    global now_img
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pf = pipeline.start(config)
    
    depth_sensor = pf.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    for t in range(20):
        frame = pipeline.wait_for_frames()
    
    while True:
        frame = pipeline.wait_for_frames()
        aligned_frames = align.process(frame)
        aligned_depth_frame = aligned_frames.get_depth_frame() 
        
        if not aligned_depth_frame:
            continue
        
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            continue
        
        depth_frame = np.asanyarray(aligned_depth_frame.get_data()).astype(np.uint8)

        depth_colormap = cv2.applyColorMap(depth_frame , cv2.COLORMAP_JET)
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite('temp_data/rgb/'+str(now_img)+'_rgb.png', color_image)
        cv2.imwrite('temp_data/depth/'+str(now_img)+'_dep.png', depth_colormap)
        img_time.append(time.time())

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
        try:
            f1 = dev.get_image(roi)
            cv2.imwrite('temp_data/tactile/'+str(now_tact)+'.png', f1)
            tact_time.append(time.time())
            now_tact += 1
        
        except:
            finger = gsdevice.Finger.MINI
            cam_id = gsdevice.get_camera_id("GelSight Mini")
            dev = gsdevice.Camera(finger, cam_id)
            dev.connect()
            
            f0 = dev.get_raw_image()


def get_data():
    global now_img, now_tact, last_img, last_tact, now_audio

    
    while img_time[last_img] + 3.0 < img_time[now_img-1]:
        # print(img_time[last_img], img_time[now_img])
        last_img += 1
    
    while tact_time[last_tact] + 3.0 < tact_time[now_tact-1]:
        last_tact += 1
    
    clip_img = (now_img - last_img)//5
    clip_tact = (now_tact - last_tact)//5
    
    
    img_arr = []
    tact_arr = []
    dep_arr = []

    for i in range(6):
        dep = Image.open('temp_data/depth/'+str(last_img + i * clip_img - (i==5)*1)+'_dep.png').convert('RGB')
        img = Image.open('temp_data/rgb/'+str(last_img + i * clip_img - (i==5)*1)+'_rgb.png').convert('RGB')
        tact = Image.open('temp_data/tactile/'+str(last_tact + i * clip_tact - (i==5)*1)+'.png').convert('RGB')

        dep_arr.append(transform(dep).unsqueeze(0).unsqueeze(0))
        img_arr.append(transform(img).unsqueeze(0).unsqueeze(0))
        tact_arr.append(transform(tact).unsqueeze(0).unsqueeze(0))
    
    depth = torch.cat(dep_arr, dim=1).permute(0, 2, 1, 3, 4)
    image = torch.cat(img_arr, dim=1).permute(0, 2, 1, 3, 4)
    tactile = torch.cat(tact_arr, dim=1).permute(0, 2, 1, 3, 4)
    
    return depth.cuda(), image.cuda(), tactile.cuda()
    
def move_arm(arm, action):
    if action == 1:
        print('forward')
        arm.set_position(x=0.75, relative=True, speed=6, wait=False)
    elif action == 2:
        print('backward')
        arm.set_position(x=-0.75, relative=True, speed=6, wait=False)
    elif action == 3:
        print('left')
        arm.set_position(y=0.75, relative=True, speed=6, wait=False)
    elif action == 4:
        print('right')
        arm.set_position(y=-0.75, relative=True, speed=6, wait=False)
    elif action == 5:
        print('down')
        arm.set_position(z=-0.75, relative=True, speed=6, wait=False)
    elif action == 6:
        print('ro left')
        arm.set_position(yaw=0.3, relative=True, speed=10, wait=False)

def main(args):
    ip = "192.168.1.224"
    arm = XArmAPI(ip, is_radian=False)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)
    arm.set_tcp_load(1.025,[0,0,73])
    arm.set_collision_sensitivity(1)
    
    seq_len = args.seq_len
    actions = torch.ones(1, seq_len, 7).cuda()
    
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
    a = model(img, img, tact, actions)

    if not os.path.exists('temp_data'):
        os.mkdir('temp_data')
    if not os.path.exists('temp_data/rgb'):
        os.mkdir('temp_data/rgb')   
    if not os.path.exists('temp_data/depth'):
        os.mkdir('temp_data/depth')   
    if not os.path.exists('temp_data/tactile'):
        os.mkdir('temp_data/tactile')   
    
    
    tti = threading.Thread(target=func_img)
    tti.start()
    
    ttt = threading.Thread(target=func_tactile)
    ttt.start()
    
    time.sleep(5)
    
    stay = 0
    his_seq = [torch.zeros(1,7)] * seq_len
    
    last_t = time.time()
    while True:
        now_t = time.time()
        if now_t - last_t >= 0.2:
            last_t = now_t
            dep, img, tact = get_data()
            
            history = torch.cat(his_seq[-seq_len:]).unsqueeze(0).cuda()

            with torch.no_grad():
                model.eval()
                out, score = model(dep.float(), img.float(), tact.float(), history.float())

            action = int(torch.argmax(out))
            onehot_action = torch.zeros(1,7)
            onehot_action[0][action] = 1.0
            his_seq.append(onehot_action)

            move_arm(arm, action)
            print(action, time.time())
            
            if(action == 0):
                stay += 1
            
            if stay >= 20:
                # task completed
                os.kill(os.getpid(), signal.SIGINT)

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)

