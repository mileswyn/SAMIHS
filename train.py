from ast import arg
import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval

from models.model_dict import get_model
from utils.data_ihs import BCIHM, Transform2D_BCIHM, Instance, Transform2D_Instance
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from tqdm import tqdm


def main():

    # ========== parameters setting ==========

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-task', required=True, default='BCIHM', help='task or dataset name')
    parser.add_argument('-sam_ckpt', required=True, type=str, default='/data/wyn/Medical-SAM-Adapter/ckpt/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('-fold', required=True, type=int, default=0, help='task or dataset name')
    parser.add_argument('--modelname', default='SAMIHS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('--encoder_input_size', type=int, default=1024, help='the image size of the encoder input, 1024 in SAM, MSA, SAMIHS, 512 in SAMUS')
    parser.add_argument('--low_image_size', type=int, default=256, help='the output image embedding size')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    # TODO
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task) 
    opt.mode = 'train'

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  ========== add the seed to make sure the results are reproducible ==========

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  ========== model and data preparation ==========
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)   
    # opt.batch_size = args.batch_size * args.n_gpu

    if args.task == 'BCIHM':
        tf_train = Transform2D_BCIHM(mode=opt.mode, img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                    p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)
        tf_val = Transform2D_BCIHM(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        train_dataset = BCIHM(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
        val_dataset = BCIHM(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    elif args.task == 'Instance':
        tf_train = Transform2D_Instance(mode=opt.mode, img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                        p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)
        tf_val = Transform2D_Instance(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        train_dataset = Instance(opt.data_path, opt.train_split, tf_train, img_size=args.encoder_input_size)
        val_dataset = Instance(opt.data_path, opt.val_split, tf_val, img_size=args.encoder_input_size)
    else:
        assert("We do not have the related dataset, please choose another task.")
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) #learning rate decay
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    # [n for n, value in model.named_parameters() if value.requires_grad == True]

    #  ========== begin to train the model ==========
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
        # ---------- Train ----------
        model.train()
        optimizer.zero_grad()
        train_losses = 0
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}', unit='img') as pbar:
            for batch_idx, (datapack) in enumerate(trainloader):
                imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
                masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
                bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
                pt = get_click_prompt(datapack, opt)
                # ---------- forward ----------
                pred = model(imgs, pt, bbox)
                train_loss = criterion(pred, masks) 
                # ---------- backward ----------
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix(**{'loss (batch)': train_loss.item()})
                train_losses += train_loss.item()
                # ---------- Adjust learning rate ----------
                if args.warmup and iter_num < args.warmup_period:
                    lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else:
                    if args.warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                        lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_
                iter_num = iter_num + 1
                pbar.update()
        scheduler.step()

        # ---------- Write log ----------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        print('lr: ', optimizer.param_groups[0]['lr'])
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        # ---------- Validation ----------
        if epoch % opt.eval_freq == 0:
            model.eval()
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, opt.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                save_path = opt.save_path + args.modelname + opt.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(round(best_dice, 4))
                torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1): # save_freq maybe no used
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()