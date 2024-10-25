import os
import csv


filelist = list(os.listdir('data/pour'))
print(len(filelist))

for now_name in filelist:
    
    actions = []
    action_times = []
    img_times = []
    tactile_times = []
    audio_begin = 0.0
    audio_end = 0.0

    f = open('data/pour/'+now_name+'/00record.txt', encoding='utf-8')
    contents = f.readlines()

    for content in contents:
        actions.append(int(content[0]))
        tt = content.find(',', 2)
        action_times.append(float(content[2:tt]))
        
    f.close()    

    f = open('data/pour/'+now_name+'/01img.txt', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        if 's' in content:
            continue
        start = content.find(',') + 1
        img_times.append(float(content[start:-1]))
    f.close()

    f = open('data/pour/'+now_name+'/02tactile.txt', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        if 's' in content:
            continue
        start = content.find(',') + 1
        tactile_times.append(float(content[start:-1]))
    f.close()

    f = open('data/pour/'+now_name+'/03audio.txt', encoding='utf-8')
    contents = f.readlines()
    audio_begin = float(contents[0])
    audio_end = float(contents[1])
    f.close()

    file = open('data/pour/'+now_name+'/data.csv', 'w', newline='')
    writer = csv.writer(file)

    now_img = 0
    last_img = 0
    now_tact = 0
    last_tact = 0
    now_audio = 0
    now_stage = 0
    his_action = []
    for t in range(len(action_times)):
        if actions[t] == 0 and t <= 10:
            continue
        if t>0 and actions[t] == 0 and actions[t-1] == 2:
            actions[t] = 2
            continue
        if t>0 and actions[t] == 0 and actions[t-1] == 3:
            actions[t] = 3

        his_action.append(actions[t])
        # print(time)
        time = action_times[t]
        for i in range(len(img_times)):
            if time - img_times[i] > 3:
                last_img = i+1
            if img_times[i] > time:
                now_img = i-1
                break
        for i in range(len(tactile_times)):
            if time - tactile_times[i] > 3:
                last_tact = i+1
            if tactile_times[i] > time:
                now_tact = i - 1
                break
        now_audio = (time - audio_begin) / (audio_end - audio_begin)
        if actions[t] == 4 and now_stage == 0:
            now_stage = 1
        if actions[t] == 0 and now_stage == 1:
            now_stage = 2
        if actions[t] == 3 and now_stage == 2:
            now_stage = 3
        
        
        if '90-' in now_name:
            now_weight = 90
        elif '120-' in now_name:
            now_weight = 120

        
        if '-60-' in now_name:
            target_weight = 60
        elif '-40-' in now_name:
            target_weight = 40
        
        writer.writerow([last_img, now_img, last_tact, now_tact, now_audio, actions[t], now_stage, his_action,now_weight,target_weight])

        
    file.close()
    print(now_name, 'ok')
