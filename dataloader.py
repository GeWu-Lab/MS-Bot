from torch.utils.data import Dataset
import os
import torchaudio
import torch
import csv
from torchvision import transforms
from PIL import Image

class ImiDataset_pour(Dataset):
    def __init__(self, mode='train', seq_len=200, soft_boundary=10):
        self.mode = mode

        self.test_list = ['120-40-1']  # change this

        self.data_list = list(os.listdir('data/pour'))
        self.train_list = []
        self.datanames = []
        for name in self.data_list:
            if name not in self.test_list:
                self.train_list.append(name)
        
        if mode == 'train':
            self.datanames = self.train_list
        else:
            self.datanames = self.test_list
            
        self.rgb_list = []
        self.tactile_list = []
        self.audio_list = []
        
        for name in self.datanames:
            self.rgb_list.append('data/'+name+'/rgb/')
            self.tactile_list.append('data/'+name+'/tactile/')
            audio,_ = torchaudio.load('data/'+name+'/audio.wav')
            audio = torchaudio.functional.resample(audio, 44100, 16000)
            self.audio_list.append(audio[0])
        
        self.name_idx = []
        self.rgb_start = []
        self.rgb_end = []
        self.tactile_start = []
        self.tactile_end = []
        self.audio_point = []
        self.label = []
        self.stage = []
        self.history = []
        self.target_weight = []
        self.init_weight = []
        
        self.label_dict = [2,-1,0,3,1]
        
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64,
            center=False,
        )
        
        for i in range(len(self.datanames)):
            csv_file = 'data/'+self.datanames[i]+'/data.csv'
            with open(csv_file, encoding='UTF-8-sig') as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    self.name_idx.append(i)
                    self.rgb_start.append(int(item[0]))
                    self.rgb_end.append(int(item[1]))
                    self.tactile_start.append(int(item[2]))
                    self.tactile_end.append(int(item[3]))
                    self.audio_point.append(float(item[4]))
                    self.label.append(int(item[5]))
                    self.stage.append(int(item[6]))
                    str_his = item[7]
                    self.init_weight.append(int(item[8]))
                    self.target_weight.append(int(item[9]))
                    
                    his = torch.zeros(seq_len, 4)
                    t = 0
                    for act in reversed(str_his):
                        if act == '[' or act == ']' or act==',' or act==' ' or t >= seq_len:
                            pass
                        else:
                            t += 1
                            his[-t][self.label_dict[int(act)]] = 1.0
                    self.history.append(his)

        self.score_weights = []
        for i in range(len(self.stage)):
            self.score_weights.append(torch.ones(4))
        for i in range(len(self.stage)):
            self.score_weights[i][self.stage[i]] = 0.0
            if i>0 and self.stage[i] != self.stage[i-1] and self.stage[i]!=0:
                for j in range(max(i-soft_boundary, 0), min(i+5, len(self.stage))):
                        self.score_weights[j][self.stage[i]] = 0.0
                        self.score_weights[j][self.stage[i-1]] = 0.0
        
        print(len(self.label))
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        name_index = self.name_idx[idx]
        rgb_start = self.rgb_start[idx]
        rgb_end = self.rgb_end[idx]
        tactile_start = self.tactile_start[idx]
        tactile_end = self.tactile_end[idx]
        audio_point = self.audio_point[idx]
        rgb_name = self.rgb_list[name_index]
        tact_name = self.tactile_list[name_index]
        
        length = self.audio_list[name_index].size()[0]
        a_time = int(audio_point * length)
        audio = self.audio_list[name_index][a_time - 16000 * 3:a_time].unsqueeze(0)
        audio = audio.view(1, audio.size()[0], 6, -1)
        # print(self.audio_list[name_index].shape)
        EPS = 1e-8
        audio = self.mel(audio.float())
        audio = torch.log(audio+ EPS)
        
        if self.mode == 'train':
            transf_img = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.RandomCrop((96, 128)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_tac = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.RandomCrop((96, 128)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
        else:
            transf_img = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.CenterCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_tac = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.CenterCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
        clip_img = (rgb_end - rgb_start)//5
        clip_tact = (tactile_end - tactile_start)//5
        img_arr = []
        tact_arr = []

        for i in range(6):
            img = Image.open(rgb_name+str(rgb_start + i * clip_img - (i==5)*1)+'.png').convert('RGB')
            tact = Image.open(tact_name+str(tactile_start + i * clip_tact - (i==5)*1)+'.png').convert('RGB')
            
            img_arr.append(transf_img(img).unsqueeze(0).unsqueeze(0))
            tact_arr.append(transf_tac(tact).unsqueeze(0).unsqueeze(0))
            
        image = torch.cat(img_arr, dim=1).permute(0, 2, 1, 3, 4)
        tactile = torch.cat(tact_arr, dim=1).permute(0, 2, 1, 3, 4)
        # a = torch.log(a+ EPS)
        return audio, image, tactile, self.label_dict[self.label[idx]], self.history[idx], self.init_weight[idx], self.target_weight[idx], self.score_weights[idx], self.stage[idx]

class ImiDataset_peg(Dataset):
    def __init__(self, mode='train', seq_len=100, soft_boundary=10):
        self.mode = mode

        self.test_list = ['1'] # change this

        self.data_list = list(os.listdir('data/peg'))
        self.train_list = []
        self.datanames = []
        for name in self.data_list:
            if name not in self.test_list:
                self.train_list.append(name)
        
        if mode == 'train':
            self.datanames = self.train_list
        else:
            self.datanames = self.test_list
            
        self.rgb_list = []
        self.dep_list = []
        self.tactile_list = []
        
        for name in self.datanames:
            self.rgb_list.append('data/peg/'+name+'/rgb/')
            self.dep_list.append('data/peg/'+name+'/depth/')
            self.tactile_list.append('data/peg/'+name+'/tactile/')
        
        self.name_idx = []
        self.rgb_start = []
        self.rgb_end = []
        self.tactile_start = []
        self.tactile_end = []
        self.label = []
        self.stage = []
        self.history = []
        
        self.rgb2_start = []
        self.rgb2_end = []
        self.audio_point = []
        sr = 16000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64,
            center=False,
        )
        
        for i in range(len(self.datanames)):
            csv_file = 'data/peg/'+self.datanames[i]+'/data.csv'
            with open(csv_file, encoding='UTF-8-sig') as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    self.name_idx.append(i)
                    self.rgb_start.append(int(item[0]))
                    self.rgb_end.append(int(item[1]))
                    self.tactile_start.append(int(item[2]))
                    self.tactile_end.append(int(item[3]))
                    self.label.append(int(item[4]))
                    str_his = item[5]
                    self.stage.append(int(item[6]))
                    
                    his = torch.zeros(seq_len, 7)
                    t = 0
                    for act in reversed(str_his):
                        if act == '[' or act == ']' or act==',' or act==' ' or t >= seq_len:
                            pass
                        else:
                            t += 1
                            his[-t][int(act)] = 1.0
                    self.history.append(his)

        self.score_weights = []
        for i in range(len(self.stage)):
            self.score_weights.append(torch.ones(3))
        for i in range(len(self.stage)):
            self.score_weights[i][self.stage[i]] = 0.0
            if i>0 and self.stage[i] != self.stage[i-1] and self.stage[i]!=0:
                for j in range(max(i-soft_boundary, 0), min(i+5, len(self.stage))):
                        self.score_weights[j][self.stage[i]] = 0.0
                        self.score_weights[j][self.stage[i-1]] = 0.0
        print(len(self.label))

        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        name_index = self.name_idx[idx]
        rgb_start = self.rgb_start[idx]
        rgb_end = self.rgb_end[idx]
        tactile_start = self.tactile_start[idx]
        tactile_end = self.tactile_end[idx]
        rgb_name = self.rgb_list[name_index]
        dep_name = self.dep_list[name_index]
        tact_name = self.tactile_list[name_index]
        
        
        if self.mode == 'train':
            transf_img = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.RandomCrop((96, 128)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.02, saturation=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_dep = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.RandomCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_tac = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.RandomCrop((96, 128)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
        else:
            transf_img = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.CenterCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_dep= transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.CenterCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
            
            transf_tac = transforms.Compose([
                    transforms.Resize((105, 140)),
                    transforms.CenterCrop((96, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                ])
        clip_img = (rgb_end - rgb_start)//5
        clip_tact = (tactile_end - tactile_start)//5
        img_arr = []
        dep_arr = []
        tact_arr = []

        for i in range(6):
            img = Image.open(rgb_name+str(rgb_start + i * clip_img - (i==5)*1)+'_rgb.png').convert('RGB')
            dep = Image.open(dep_name+str(rgb_start + i * clip_img - (i==5)*1)+'_dep.png').convert('RGB')
            tact = Image.open(tact_name+str(tactile_start + i * clip_tact - (i==5)*1)+'.png').convert('RGB')
            
            img_arr.append(transf_img(img).unsqueeze(0).unsqueeze(0))
            dep_arr.append(transf_dep(dep).unsqueeze(0).unsqueeze(0))
            tact_arr.append(transf_tac(tact).unsqueeze(0).unsqueeze(0))
            
        depth = torch.cat(dep_arr, dim=1).permute(0, 2, 1, 3, 4)
        image = torch.cat(img_arr, dim=1).permute(0, 2, 1, 3, 4)
        tactile = torch.cat(tact_arr, dim=1).permute(0, 2, 1, 3, 4)

        return depth, image, tactile, self.label[idx], self.history[idx], self.score_weights[idx], self.stage[idx]