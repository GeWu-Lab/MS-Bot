import os
import csv


filelist = list(os.listdir('data/peg'))
print(len(filelist))

for now_name in filelist:
    
    actions = []
    actions_his = []
    action_times = []
    img_times = []
    tactile_times = []

    f = open('data/peg/'+now_name+'/00record.txt', encoding='utf-8')
    contents = f.readlines()

    all_prop = []
    
    for content in contents:
        actions.append(int(content[0]))
        tt = content.find(',', 2)
        action_times.append(float(content[2:tt]))
        bb = tt+2
        prop = []
        for i in range(tt+2, len(content)):
            if content[i] == ',' or content[i] == ']':
                prop.append(float(content[bb:i]))
                bb = i+1
        all_prop.append(prop)
        
    f.close()    

    f = open('data/peg/'+now_name+'/01img.txt', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        if 's' in content:
            continue
        start = content.find(',') + 1
        img_times.append(float(content[start:-1]))
    f.close()

    f = open('data/peg/'+now_name+'/02tactile.txt', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        if 's' in content:
            continue
        start = content.find(',') + 1
        tactile_times.append(float(content[start:-1]))
    f.close()

    file = open('data/peg/'+now_name+'/data.csv', 'w', newline='')
    writer = csv.writer(file)

    now_img = 0
    last_img = 0
    now_tact = 0
    last_tact = 0
    now_audio = 0
    now_stage = 0
    last_flag=False
    has_rotate=False
    now_stage = 0
    for t in range(len(action_times)):  
        if actions[t] == 0:
            flag = False
            next_act = -1
            for a_index in range(t, len(action_times)):
                if actions[a_index] != 0:
                    next_act = actions[a_index]
                    flag=True
                    break
            
            if flag:
                actions[t] = next_act
                if not last_flag:
                    actions_his.append(actions[t-1])
                last_flag=True
                
            else:
                if t>0:
                    actions_his.append(actions[t-1])

        else:
            if t>0 and last_flag==False:
                actions_his.append(actions[t-1])
            else:
                last_flag=False

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
        if actions[t] == 6:
            has_rotate = True
        
        if not has_rotate:
            now_stage = 0
            
        else:
            now_stage = 2
            for pp in range(t, len(actions)):
                if actions[pp] == 6:
                    now_stage = 1
                    break

        writer.writerow([last_img, now_img, last_tact, now_tact, actions[t], actions_his, now_stage])

        
    file.close()
    print(now_name, 'ok')
