from dataloader import ImiDataset_pour
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
import random
from models.model_pour import ConcatModel, MULSA, MSBot
import os
from config import parse_args

stage_num = 4

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_tokens(args, model, dataloader):
    all_num = [0,0,0]
    all_token = torch.zeros((stage_num, 128)).cuda()
    with torch.no_grad():
        model.eval()
        for step, (spec, image, tactile, label, actions, init_weight, target_weight, score_weight, stage) in enumerate(dataloader):
            spec = spec.squeeze(1).cuda()
            image = image.squeeze(1).cuda()
            tactile = tactile.squeeze(1).cuda()
            actions = actions.squeeze(1).cuda()

            if args.pour_setting == 'init':
                weight_prompt = init_weight.int().cuda()
            elif args.pour_setting == 'target':
                weight_prompt = target_weight.int().cuda()
            
            state_tokens = model.forward_state(spec.float(), image.float(), tactile.float(), actions.float(), weight_prompt)
            state_tokens = state_tokens.squeeze(1)
            for i in range(len(stage)):
                all_token[stage[i]] += state_tokens[i]
                all_num[stage[i]] += 1.0
                
        for i in range(stage_num):
            all_token[i] /= all_num[i]
        
        model.all_stage_tokens.data=all_token
    
    print('Stage Tokens init success')

def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_score = 0
    warmup_flag = False
    if epoch < args.warmup_epochs:
        warmup_flag=True

    for step, (spec, image, tactile, label, actions, init_weight, target_weight, score_weight, stage) in enumerate(dataloader):

        spec = spec.squeeze(1).cuda()
        image = image.squeeze(1).cuda()
        tactile = tactile.squeeze(1).cuda()
        actions = actions.squeeze(1).cuda()

        if args.pour_setting == 'init':
            weight_prompt = init_weight.int().cuda()
        elif args.pour_setting == 'target':
            weight_prompt = target_weight.int().cuda()

        score_weight = score_weight.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        out, score = model(spec.float(), image.float(), tactile.float(), actions.float(), weight_prompt, warmup=warmup_flag)

        if warmup_flag:
            loss = criterion(out, label)
        else:
            if args.model == 'MSBot':
                loss_score = args.penalty_intensity * torch.mean(score * score_weight)
            else:
                loss_score = torch.zeros(1).cuda()
            _loss_score += loss_score.item()
            loss = criterion(out, label) + loss_score

        loss.backward()
        optimizer.step()

        _loss += loss.item()

    scheduler.step()
    print('Loss Score:', _loss_score / len(dataloader))

    return _loss / len(dataloader)
    
def valid(args, model, dataloader, epoch):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    n_classes = 5
    _loss = 0

    warmup_flag = False
    if epoch < args.warmup_epochs:
        warmup_flag=True

    with torch.no_grad():
        model.eval()

        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, tactile, label, actions, init_weight, target_weight, score_weight, stage) in enumerate(dataloader):

            spec = spec.squeeze(1).cuda()
            image = image.squeeze(1).cuda()
            tactile = tactile.squeeze(1).cuda()
            actions = actions.squeeze(1).cuda()

            if args.pour_setting == 'init':
                weight_prompt = init_weight.int().cuda()
            elif args.pour_setting == 'target':
                weight_prompt = target_weight.int().cuda()

            out, score = model(spec.float(), image.float(), tactile.float(), actions.float(), weight_prompt, warmup=warmup_flag)
            loss = criterion(out, label.cuda())
            _loss += loss.item()
            prediction = softmax(out)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if label[i] == ma:
                    acc[label[i]] += 1.0

    print("Val Loss: {:.4f}, Acc 0: {:.4f}, Acc 2: {:.4f}, Acc 3: {:.4f}, Acc 4: {:.4f}".format(_loss / len(dataloader)
                                                                                            , acc[0]/num[0], acc[2]/num[2]
                                                                                            , acc[3]/num[3], acc[4]/num[4]))
    return sum(acc) / sum(num), _loss / len(dataloader)

def main(args):
    setup_seed(args.seed)

    if args.model == 'Concat':
        model = ConcatModel().cuda()
    elif args.model == 'MULSA':
        model = MULSA().cuda()
    elif args.model == 'MSBot':
        model = MSBot().cuda()
    else:
        print('Model not found')
        exit(0) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.9
    )
    
    train_dataset = ImiDataset_pour(mode='train', seq_len=args.seq_len, soft_boundary=args.gamma)
    test_dataset = ImiDataset_pour(mode='test', seq_len=args.seq_len, soft_boundary=args.gamma)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    EPOCHS = args.epochs

    best_acc = 0.0
    best_vloss = 100

    if os.path.exists('checkpoints') == False:
        os.mkdir('checkpoints')

    if os.path.exists('checkpoints/pour') == False:
        os.mkdir('checkpoints/pour')
    
    if args.model_dir != None and args.model_dir != '':
        ckpt_dir = 'checkpoints/pour/'+args.model_dir+'/'
    else:
        ckpt_dir = 'checkpoints/pour/'+args.model+'/'
    if os.path.exists('ckpt_dir') == False:
        os.mkdir(ckpt_dir)

    for epoch in range(EPOCHS):
        
        if epoch == args.warmup_epochs and args.model == 'MSBot':
            set_tokens(args, model, train_dataloader)

        print('Epoch: {}: '.format(epoch))
        batch_loss = train_epoch(args, epoch, model, train_dataloader, optimizer, scheduler)
        acc, vloss = valid(args, model, test_dataloader, epoch)
        
        if vloss < best_vloss:
            best_vloss = float(vloss)
            
            saved_dict = {'model': model.state_dict()}
            model_name = ckpt_dir+'model_'+str(epoch)+'_vloss.pth'
            torch.save(saved_dict, model_name)
            print('The model has been saved at ' + model_name)
        
        if acc > best_acc:
            best_acc = float(acc)
            
            saved_dict = {'model': model.state_dict()}
            model_name = ckpt_dir+'model_'+str(epoch)+'_acc.pth'
            torch.save(saved_dict, model_name)
            print('The model has been saved at ' + model_name)
        
            print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
        else:
            print("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}".format(batch_loss, acc, best_acc))

if __name__ == "__main__":
    args = parse_args()
    args = args.parse_args()
    main(args)
        