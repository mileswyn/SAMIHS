# this file is utilized to evaluate the models from mode: 2D-slice level
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
from utils.generate_prompts import get_click_prompt
import time
import pandas as pd
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as Func
from tqdm import tqdm
import cv2

def eval_mask_slice2(valloader, model, criterion, opt, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    fore_dice = []
    hds_fore = []
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    with tqdm(total=len(valloader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch_idx, (datapack) in enumerate(valloader):
            imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
            masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
            label = Variable(datapack['label'].to(dtype = torch.float32, device=opt.device))
            bbox = Variable(datapack['bbox'].to(dtype = torch.float32, device=opt.device))
            class_id = datapack['class_id']
            image_filename = datapack['image_name']

            pt = get_click_prompt(datapack, opt)

            with torch.no_grad():
                start_time = time.time()
                pred = model(imgs, pt, bbox=None)
                sum_time =  sum_time + (time.time()-start_time)

            val_loss = criterion(pred, masks)
            pbar.set_postfix(**{'loss (batch)': val_loss.item()})
            val_losses += val_loss.item()

            gt = label.detach().cpu().numpy()
            
            gt = gt[:, 0, :, :]
            predict = torch.sigmoid(pred['masks'])
            predict = Func.resize(predict, (512, 512), InterpolationMode.BILINEAR)
            predict = predict.detach().cpu().numpy()  # (b, c, h, w)
            seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
            b, h, w = seg.shape
            for j in range(0, b):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1, :, :] == 1] = 1
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1, :, :] == 1] = 1
                dice_i = metrics.dice_coefficient(pred_i, gt_i)
                #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
                dices[eval_number+j, 1] += dice_i
                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
                ious[eval_number+j, 1] += iou
                accs[eval_number+j, 1] += acc
                ses[eval_number+j, 1] += se
                sps[eval_number+j, 1] += sp
                hds_i = hausdorff_distance(pred_i[0, :, :].astype(np.float32), gt_i[0, :, :].astype(np.float32))
                hds[eval_number+j, 1] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :])
                if gt_i.sum() > 0:
                    fore_dice.append(dice_i)
                    # if opt.visual:
                        # cv2.imwrite(os.path.join('/SAMIHS/valid_output/', image_filename[0].split('.png')[0] +'_' + args.modelname + '_' + str(round(dice_i,3)) + '_' + str(round(hds_i,3)) + '.png'), pred_i[0]*255)
                        # cv2.imwrite(os.path.join('/SAMIHS/valid_output/', image_filename[0].split('.png')[0] +'_gt_' + args.modelname + '.png'), gt_i[0]*255)
                    hds_fore.append(hausdorff_distance(pred_i[0, :, :].astype(np.float32), gt_i[0, :, :].astype(np.float32)))
                del pred_i, gt_i
            eval_number = eval_number + b
            pbar.update()
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
    val_losses = val_losses / (batch_idx + 1)

    fore_dice_mean = np.mean(np.array(fore_dice))
    hds_fore_mean = np.mean(np.array(hds_fore))
    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])
    mean_hdis = np.mean(hd_mean[1:])
    print("test speed", eval_number/sum_time)
    if opt.mode == "train":
        return dices, fore_dice_mean, hds_fore_mean, val_losses
    else:
        # data = pd.DataFrame(dices*100)
        # writer = pd.ExcelWriter('./result/' + args.task + '/PT10-' + opt.modelname + '.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.2f')
        # writer._save()
        fore_dice_mean = np.mean(np.array(fore_dice)*100)
        fore_dice_std = np.std(np.array(fore_dice)*100)
        fore_hds_mean = np.mean(np.array(hds_fore))
        fore_hds_std = np.std(np.array(hds_fore))
        dice_mean = np.mean(dices*100, axis=0)
        dices_std = np.std(dices*100, axis=0)
        hd_mean = np.mean(hds, axis=0)
        hd_std = np.std(hds, axis=0)
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        # print(hds_fore)
        return fore_dice_mean, fore_hds_mean, iou_mean, acc_mean, se_mean, sp_mean, fore_dice_std, fore_hds_std, iou_std, acc_std, se_std, sp_std

def get_eval(valloader, model, criterion, opt, args):
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)