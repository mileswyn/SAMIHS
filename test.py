from ast import arg
import os
import argparse
import torch
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
from thop import profile


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('-task', required=True, default='BCIHM', help='task or dataset name')
    parser.add_argument('-sam_ckpt', required=True, type=str, default='/data/wyn/Medical-SAM-Adapter/ckpt/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('-fold', required=True, type=int, default=0, help='task or dataset name')
    parser.add_argument('-encoder_input_size', type=int, default=1024, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=256, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--modelname', default='SAMIHS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)
    print("task", args.task, "checkpoints:", opt.load_path)
    opt.mode = "test"
    opt.visual = False
    opt.modelname = args.modelname
    device = torch.device(opt.device)

     #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    # opt.batch_size = args.batch_size * args.n_gpu
    if args.task == 'BCIHM':
        tf_val = Transform2D_BCIHM(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        val_dataset = BCIHM(opt.data_path, opt.test_split, tf_val, fold=args.fold, img_size=args.encoder_input_size, class_id=1)
    elif args.task == 'Instance':
        tf_val = Transform2D_Instance(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        val_dataset = Instance(opt.data_path, opt.val_split, tf_val, fold=args.fold, img_size=args.encoder_input_size)
    else:
        assert("We do not have the related dataset, please choose another task.")
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

#  ========== begin to evaluate the model ==========

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("mean dice:", mean_dice)
    else:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        print("dataset:" + args.task + " -----------model name: "+ args.modelname)
        print("task", args.task, "checkpoints:", opt.load_path)
        print(mean_dice, mean_hdis, mean_iou[1:], mean_acc[1:], mean_se[1:], mean_sp[1:])
        print(std_dice, std_hdis, std_iou[1:], std_acc[1:], std_se[1:], std_sp[1:])
        # with open("experiments.txt", "a+") as file:
        #     file.write(args.task + " " + args.modelname + "-pt10 " + '%.2f'%(mean_dice[1]) + "±" + '%.2f'%std_dice[1] + " ")
        #     file.write('%.2f'%mean_hdis[1] + "±" + '%.2f'%std_hdis[1] + " ")
        #     file.write('%.2f'%(mean_iou[1]) + "±" + '%.2f'%std_iou[1] + " ")
        #     file.write('%.2f'%(mean_acc[1]) + "±" + '%.2f'%std_acc[1] + " ")
        #     file.close()
    
if __name__ == '__main__':
    main()
