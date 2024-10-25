import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet18, Attention, CrossAttention

class ConcatModel(nn.Module):
    def __init__(self):
        super(ConcatModel, self).__init__()

        self.depth_net = resnet18(modality='visual')
        self.visual_net = resnet18(modality='visual')
        self.tact_net = resnet18(modality='tactile')
        
        loaded_dict = torch.load('resnet18.pth')
        self.depth_net.load_state_dict(loaded_dict, strict=False)
        self.visual_net.load_state_dict(loaded_dict, strict=False)
        self.tact_net.load_state_dict(loaded_dict, strict=False)
        del loaded_dict
        
        
        self.depth_proj = nn.Linear(512, 128)
        self.visual_proj = nn.Linear(512, 128)
        self.tact_proj = nn.Linear(512, 128)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 3, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 7),
        )

    def forward(self, depth, visual, tactile, actions=None):

        EPS = 1e-8
        B = visual.size()[0]
        a = self.depth_net(depth)
        
        v = self.visual_net(visual)
        t = self.tact_net(tactile)

        (_, C, H, W) = a.size()
        a = a.view(B, -1, C, H, W)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        (_, C, H, W) = t.size()
        t = t.view(B, -1, C, H, W)


        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool2d(v, 1)
        t = F.adaptive_avg_pool2d(t, 1)

        a = torch.flatten(a, 2)
        v = torch.flatten(v, 2)
        t = torch.flatten(t, 2)
        
        a = self.depth_proj(a)
        v = self.visual_proj(v)
        t = self.tact_proj(t)
        
        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        t = torch.flatten(t, 1)
        
        out = torch.cat([a,v,t], dim=1)
        
        out = torch.flatten(out, 1)
        out = self.mlp(out)
        

        return out, torch.zeros_like(out)

class MULSA(nn.Module):
    def __init__(self):
        super(MULSA, self).__init__()

        self.depth_net = resnet18(modality='visual')
        self.visual_net = resnet18(modality='visual')
        self.tact_net = resnet18(modality='tactile')
        
        loaded_dict = torch.load('resnet18.pth')
        self.depth_net.load_state_dict(loaded_dict, strict=False)
        self.visual_net.load_state_dict(loaded_dict, strict=False)
        self.tact_net.load_state_dict(loaded_dict, strict=False)
        del loaded_dict
        
        self.depth_proj = nn.Linear(512, 128)
        self.visual_proj = nn.Linear(512, 128)
        self.tact_proj = nn.Linear(512, 128)
        
        self.mha = Attention(128, 8)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 3, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 7),
        )

    def forward(self, depth, visual, tactile, actions=None):

        EPS = 1e-8
        B = visual.size()[0]
        # a = self.mel(depth.float())
        # a = torch.log(a+ EPS)
        # print(a.shape)
        a = self.depth_net(depth)
        
        v = self.visual_net(visual)
        t = self.tact_net(tactile)

        (_, C, H, W) = a.size()
        a = a.view(B, -1, C, H, W)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        (_, C, H, W) = t.size()
        t = t.view(B, -1, C, H, W)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool2d(v, 1)
        t = F.adaptive_avg_pool2d(t, 1)

        a = torch.flatten(a, 2)
        v = torch.flatten(v, 2)
        t = torch.flatten(t, 2)
        
        a = self.depth_proj(a)
        v = self.visual_proj(v)
        t = self.tact_proj(t)

        all = torch.cat([a,v,t], dim=1)
        out, attn = self.mha(all)
        
        out += all
        
        out = torch.flatten(out, 1)
        out = self.mlp(out)

        return out, torch.zeros_like(out)

    
    
class MSBot(nn.Module):
    def __init__(self, blur_p = 0.25, beta = 0.5):
        super(MSBot, self).__init__()

        self.depth_net = resnet18(modality='visual')
        self.visual_net = resnet18(modality='visual')
        self.tact_net = resnet18(modality='tactile')
        
        loaded_dict = torch.load('resnet18.pth')
        self.depth_net.load_state_dict(loaded_dict, strict=False)
        self.visual_net.load_state_dict(loaded_dict, strict=False)
        self.tact_net.load_state_dict(loaded_dict, strict=False)
        del loaded_dict
        
        self.depth_proj = nn.Linear(512, 128)
        self.visual_proj = nn.Linear(512, 128)
        self.tact_proj = nn.Linear(512, 128)
        
        self.mha = CrossAttention(128, 8, blur_p=blur_p)
        self.state_tokenizer = nn.Linear(128*4, 128)

        self.beta = beta

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 7),
        )
        
        self.lstm = torch.nn.LSTM(7, 128, 1, batch_first=True)
        
        stage = 3
        self.all_stage_tokens = torch.nn.Parameter(torch.zeros(stage,128), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.gate = torch.nn.Linear(128,3)
        
    def forward_state(self, depth, visual, tactile, actions):
        EPS = 1e-8
        B = visual.size()[0]

        a = self.depth_net(depth)
        v = self.visual_net(visual)
        t = self.tact_net(tactile)

        (_, C, H, W) = a.size()
        a = a.view(B, -1, C, H, W)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        (_, C, H, W) = t.size()
        t = t.view(B, -1, C, H, W)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool2d(v, 1)
        t = F.adaptive_avg_pool2d(t, 1)

        a = torch.flatten(a, 2)
        v = torch.flatten(v, 2)
        t = torch.flatten(t, 2)
        
        a = self.depth_proj(a)
        v = self.visual_proj(v)
        t = self.tact_proj(t)
        
        
        h, _ = self.lstm(actions)
        h = h[:, -1].unsqueeze(1)

        state_token = torch.cat([a[:,-1],v[:,-1],t[:,-1],h.squeeze(1)], dim=1).flatten(1)
        state_token = self.state_tokenizer(state_token).unsqueeze(1)
        
        return state_token

    def forward(self, depth, visual, tactile, actions, warmup=False, return_attn=False):

        B = visual.size()[0]

        a = self.depth_net(depth)
        v = self.visual_net(visual)
        t = self.tact_net(tactile)

        (_, C, H, W) = a.size()
        a = a.view(B, -1, C, H, W)
        
        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        
        (_, C, H, W) = t.size()
        t = t.view(B, -1, C, H, W)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool2d(v, 1)
        t = F.adaptive_avg_pool2d(t, 1)

        a = torch.flatten(a, 2)
        v = torch.flatten(v, 2)
        t = torch.flatten(t, 2)
        
        a = self.depth_proj(a)
        v = self.visual_proj(v)
        t = self.tact_proj(t)
        
        
        h, _ = self.lstm(actions)
        h = h[:, -1].unsqueeze(1)

        state_token = torch.cat([a[:,-1],v[:,-1],t[:,-1],h.squeeze(1)], dim=1).flatten(1)
        state_token = self.state_tokenizer(state_token).unsqueeze(1)

        score = 0.0
        if not warmup:
            score = self.gate(state_token.squeeze(1).detach())
            score = self.softmax(score)
            
            state_token = self.beta * state_token + (1-self.beta) * torch.mm(score , self.all_stage_tokens).unsqueeze(1)


        
        all = torch.cat([a,v,t], dim=1)
        out, attn = self.mha(state_token,all)
        

        out = torch.flatten(out, 1)

        out = self.mlp(out)

        return out, score
    