import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from adabelief_pytorch import AdaBelief
import itertools
import time
import os
import copy
import math

from utils import define_G, define_D, ImagePool, GANLoss, get_scheduler, print_network, U_Net, calc_loss, define_F, PatchNCELoss

class Solver(nn.Module):
    def __init__(self, opts, cuda=None):
        super().__init__()
        self.opts=opts
        self.tensorboardWriter = SummaryWriter('./log_version_0.0')
        self.writerEpoch = 0
        self.gpu_ids = cuda

        self.use_sigmoid = False
        self.netG_AB = define_G(input_nc=1, output_nc=1, ngf=32, which_model_netG='resnet_6blocks', norm='instance', use_dropout=False, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netG_BA = define_G(input_nc=1, output_nc=1, ngf=32, which_model_netG='resnet_6blocks', norm='instance', use_dropout=False, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netD_B = define_D(input_nc=1, ndf=32, which_model_netD='basic', n_layers_D=3, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netD_gc_B = define_D(input_nc=1, ndf=32, which_model_netD='basic', n_layers_D=3, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netD_A = define_D(input_nc=1, ndf=32, which_model_netD='basic', n_layers_D=3, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netD_gc_A = define_D(input_nc=1, ndf=32, which_model_netD='basic', n_layers_D=3, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])

        self.segnet_ab = U_Net(in_ch=1, out_ch=1)
        self.segnet_ba = U_Net(in_ch=1, out_ch=1)
        self.netD_seg = define_D(input_nc=1, ndf=16, which_model_netD='basic', n_layers_D=2, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])
        self.netD_seg_gc = define_D(input_nc=1, ndf=16, which_model_netD='basic', n_layers_D=2, norm='instance', use_sigmoid=self.use_sigmoid, init_type='xavier', gpu_ids=[self.gpu_ids])

        self.netF_AB = define_F(input_nc=1, netF='mlp_sample', norm='instance', use_dropout=False, init_type='xavier', init_gain=0.02, no_antialias=True, gpu_ids=[self.gpu_ids], netF_nc=256)
        self.netF_BA = define_F(input_nc=1, netF='mlp_sample', norm='instance', use_dropout=False, init_type='xavier', init_gain=0.02, no_antialias=True, gpu_ids=[self.gpu_ids], netF_nc=256)
        self.nce_layers = [0,4,8,12,16]
        self.lambda_NCE = 1.0
        self.num_patchs = 256
        self.cur_epoch = 0
        self.criterionNCE = []
        self.nce_T = 0.07  # temperature for NCE loss
        nce_includes_all_negatives_from_minibatch = False
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(batch_size=self.opts.bs, nce_T=self.nce_T, nce_includes_all_negatives_from_minibatch=nce_includes_all_negatives_from_minibatch).cuda(self.gpu_ids))

        self.fake_B_pool = ImagePool(pool_size=50)
        self.fake_gc_B_pool = ImagePool(pool_size=50)
        self.fake_A_pool = ImagePool(pool_size=50)
        self.fake_gc_A_pool = ImagePool(pool_size=50)
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=self.Tensor, cuda=self.gpu_ids).cuda(self.gpu_ids)
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionGc = torch.nn.L1Loss()
        self.criterionCycle = torch.nn.L1Loss()

        # lambda
        self.lambda_G = 1.0
        self.lambda_gc = 2.0
        self.lambda_AB = 10.0
        # identity
        self.identity = 0.5
        # rot90
        self.fineSize = 512

        # initialize optimizers
        self.seg_ab_opt = torch.optim.Adam(self.segnet_ab.parameters(), lr=1e-4)
        self.seg_ba_opt = torch.optim.Adam(self.segnet_ba.parameters(), lr=1e-4)
        self.seg_ab_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.seg_ab_opt, int(1e10), eta_min=1e-5)
        self.seg_ba_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.seg_ba_opt, int(1e10), eta_min=1e-5)

        self.optimizer_D_seg = AdaBelief(itertools.chain(self.netD_seg.parameters(), self.netD_seg_gc.parameters()), lr=2e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
        self.optimizer_G = AdaBelief(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters(),), lr=1e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
        self.optimizer_D_B = AdaBelief(itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=2e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
        self.optimizer_D_A = AdaBelief(itertools.chain(self.netD_A.parameters(), self.netD_gc_A.parameters()), lr=2e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_B)
        self.optimizers.append(self.optimizer_D_A)
        self.optimizers.append(self.optimizer_D_seg)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer))
        
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test_seg(self, b):
        with torch.no_grad():
            self.test_segment = self.segnet_ab(b)
        return self.test_segment
    
    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.fineSize
        inv_idx = torch.arange(size-1, -1, -1).long().cuda(self.gpu_ids)
        if direction == 0:
          tensor = torch.index_select(tensor, 3, inv_idx)
        else:
          tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor
    
    def gc_forward(self):
        input_A = self.x_a.clone()
        input_B = self.x_b.clone()
        input_A_gt = self.x_a_GT.clone()

        self.real_A = self.x_a
        self.real_B = self.x_b

        self.real_gc_A = self.rot90(input_A, 0)
        self.real_gc_B = self.rot90(input_B, 0)
        self.real_gc_A_gt = self.rot90(input_A_gt, 0)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0
        if direction == 0:
          AB_gt = self.rot90(AB_gc.clone().detach(), 1)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 0)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
          AB_gt = self.rot90(AB_gc.clone().detach(), 0)
          loss_gc = self.criterionGc(AB, AB_gt)
          AB_gc_gt = self.rot90(AB.clone().detach(), 1)
          loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        loss_gc = loss_gc*self.lambda_AB*self.lambda_gc
        return loss_gc
    
    def initialize_NCE(self,x_a,x_b,x_a_GT):
        self.x_a = x_a
        self.x_b = x_b
        self.x_a_GT = x_a_GT
        self.gc_forward()
        self.backward_G()
        self.backward_D_B()
        self.backward_D_A()
        self.backward_D_seg()
        if self.lambda_NCE > 0.0:
            self.optimizer_F = AdaBelief(itertools.chain(self.netF_AB.parameters(), self.netF_BA.parameters(),), 
                                         lr=1e-4, weight_decay=0,eps=1e-16, betas=(0.5, 0.9), weight_decouple=True, rectify=True, print_change_log=False)
            self.optimizers.append(self.optimizer_F)
            self.schedulers.append(get_scheduler(self.optimizer_F))

    def calculate_NCE_loss(self, src, tgt, src_mask=None, direction='AB'):
        n_layers = len(self.nce_layers)
        if src_mask is not None:
            feat_k_mask = self.netG_AB(src_mask, self.nce_layers, encode_only=True)
        if direction == 'AB':
            feat_q = self.netG_AB(tgt, self.nce_layers, encode_only=True)
            feat_k = self.netG_AB(src, self.nce_layers, encode_only=True)

            feat_k_pool, sample_ids = self.netF_AB(feat_k, self.num_patchs, None, feat_k_mask)
            feat_q_pool, _ = self.netF_AB(feat_q, self.num_patchs, sample_ids, None)
        elif direction == 'BA':
            feat_q = self.netG_BA(tgt, self.nce_layers, encode_only=True)
            feat_k = self.netG_BA(src, self.nce_layers, encode_only=True)

            feat_k_pool, sample_ids = self.netF_BA(feat_k, self.num_patchs, None, feat_k_mask)
            feat_q_pool, _ = self.netF_BA(feat_q, self.num_patchs, sample_ids, None)
        else:
            raise Exception("not right direction!")

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def backward_G(self):
        self.real_A_mask = self.segnet_ba((self.real_A+1.0)*0.5)
        loss_seg_BA = calc_loss(self.real_A_mask, self.x_a_GT)
        self.real_A_gc_mask = self.segnet_ba((self.real_gc_A+1.0)*0.5)
        loss_seg_BA += calc_loss(self.real_A_gc_mask, self.real_gc_A_gt)

        fake_B = self.netG_AB.forward(self.real_A)
        self.x_ab_ori = fake_B
        self.fake_B_mask = self.segnet_ab((fake_B+1.0)*0.5)
        loss_seg_AB = calc_loss(self.fake_B_mask, self.x_a_GT)

        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.lambda_G

        fake_gc_B = self.netG_AB.forward(self.real_gc_A)
        self.x_gc_ab_ori = fake_gc_B
        self.fake_gc_B_mask = self.segnet_ab((fake_gc_B+1.0)*0.5)
        loss_seg_AB += calc_loss(self.fake_gc_B_mask, self.real_gc_A_gt)

        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.lambda_G

        self.real_B_mask = self.segnet_ab((self.real_B+1.0)*0.5)
        pred_fake = self.netD_seg.forward(self.real_B_mask)
        loss_G_seg = self.criterionGAN(pred_fake, True) * self.lambda_G

        fake_A = self.netG_BA.forward(self.real_B)
        self.fake_A_ori = fake_A
        pred_fake = self.netD_A.forward(fake_A)
        loss_G_AB += self.criterionGAN(pred_fake, True) * self.lambda_G

        self.real_gc_B_mask = self.segnet_ab((self.real_gc_B+1.0)*0.5)
        pred_fake = self.netD_seg_gc.forward(self.real_gc_B_mask)
        loss_G_seg += self.criterionGAN(pred_fake, True) * self.lambda_G

        fake_gc_A = self.netG_BA.forward(self.real_gc_B)
        pred_fake = self.netD_gc_A.forward(fake_gc_A)
        loss_G_gc_AB += self.criterionGAN(pred_fake, True) * self.lambda_G

        loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        loss_gc += self.get_gc_rot_loss(fake_A, fake_gc_A, 0)

        if self.cur_epoch ==0 or self.cur_epoch > 100:
            loss_NCE_B = self.calculate_NCE_loss(self.real_A, fake_B, self.real_A_mask, direction='AB')
            loss_NCE_B += self.calculate_NCE_loss(self.real_gc_A, fake_gc_B, self.real_A_gc_mask, direction='AB')
            loss_NCE_A = self.calculate_NCE_loss(self.real_B, fake_A, self.real_B_mask, direction='BA')
            loss_NCE_A += self.calculate_NCE_loss(self.real_gc_B, fake_gc_A, self.real_gc_B_mask, direction='BA')
            loss_NCE_both = (loss_NCE_B + loss_NCE_A) * 0.25
            self.tensorboardWriter.add_scalar('loss_NCE_B', loss_NCE_B, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_NCE_A', loss_NCE_A, self.writerEpoch)
        else:
            loss_NCE_both = 0

        if self.identity > 0:
            idt_A = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.lambda_AB * self.identity
            idt_gc_A = self.netG_AB(self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.lambda_AB * self.identity

            idt_B = self.netG_BA(self.real_A)
            loss_idt += self.criterionIdt(idt_B, self.real_A) * self.lambda_AB * self.identity
            idt_gc_B = self.netG_BA(self.real_gc_A)
            loss_idt_gc += self.criterionIdt(idt_gc_B, self.real_gc_A) * self.lambda_AB * self.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        self.tensorboardWriter.add_scalar('loss_G_AB', loss_G_AB, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_G_gc_AB', loss_G_gc_AB, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_gc', loss_gc, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_idt', loss_idt, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_idt_gc', loss_idt_gc, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_seg_AB', loss_seg_AB, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_seg_BA', loss_seg_BA, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_G_seg', loss_G_seg, self.writerEpoch)

        rec_A = self.netG_BA(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * self.lambda_AB
        rec_gc_A = self.netG_BA(fake_gc_B)
        loss_cycle_A += self.criterionCycle(rec_gc_A, self.real_gc_A) * self.lambda_AB

        rec_B = self.netG_AB(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * self.lambda_AB
        rec_gc_B = self.netG_AB(fake_gc_A)
        loss_cycle_B += self.criterionCycle(rec_gc_B, self.real_gc_B) * self.lambda_AB

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc + \
                 loss_cycle_A + loss_cycle_B + loss_seg_AB + loss_seg_BA + loss_G_seg + loss_NCE_both

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data
        self.fake_A = fake_A.data
        self.fake_gc_A = fake_gc_A.data

    def backward_D_basic(self, netD, real, fake, netD_gc, real_gc, fake_gc, lossA=False):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, True)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, False)
        # Combined loss
        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

        if lossA:    
            self.tensorboardWriter.add_scalar('loss_D_real_A', loss_D_real, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_fake_A', loss_D_fake, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_gc_real_A', loss_D_gc_real, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_gc_fake_A', loss_D_gc_fake, self.writerEpoch)
        else:
            self.tensorboardWriter.add_scalar('loss_D_real_B', loss_D_real, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_fake_B', loss_D_fake, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_gc_real_B', loss_D_gc_real, self.writerEpoch)
            self.tensorboardWriter.add_scalar('loss_D_gc_fake_B', loss_D_gc_fake, self.writerEpoch)

        # backward
        loss_D.backward()
        return loss_D
    
    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_gc_B, self.real_gc_B, fake_gc_B)

    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        fake_gc_A = self.fake_gc_A_pool.query(self.fake_gc_A)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A, self.netD_gc_A, self.real_gc_A, fake_gc_A, lossA=True)

    def backward_D_seg(self):
        self.real_A_mask = self.segnet_ba((self.real_A+1.0)*0.5)
        pred_real = self.netD_seg(self.real_A_mask)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = self.netD_seg(self.real_B_mask.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # gc
        self.real_A_gc_mask = self.segnet_ba((self.real_gc_A+1.0)*0.5)
        pred_real = self.netD_seg_gc(self.real_A_gc_mask)
        loss_D_gc_real = self.criterionGAN(pred_real, True)

        pred_fake = self.netD_seg_gc(self.real_gc_B_mask.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake, False)

        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

        self.tensorboardWriter.add_scalar('loss_D_real_seg', loss_D_real, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_D_fake_seg', loss_D_fake, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_D_gc_real_seg', loss_D_gc_real, self.writerEpoch)
        self.tensorboardWriter.add_scalar('loss_D_gc_fake_seg', loss_D_gc_fake, self.writerEpoch)
        loss_D.backward()

    def gan_forward(self,x_a,x_b,x_a_GT,epoch):
        self.cur_epoch = epoch
        self.x_a = x_a
        self.x_b = x_b
        self.x_a_GT = x_a_GT
        self.gc_forward()

        self.optimizer_G.zero_grad()
        if self.cur_epoch == 0 or self.cur_epoch > 100:
            self.optimizer_F.zero_grad()
        self.seg_ba_opt.zero_grad()
        self.seg_ab_opt.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.cur_epoch == 0 or self.cur_epoch > 100:
            self.optimizer_F.step()
        self.seg_ba_opt.step()
        self.seg_ab_opt.step()

        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        self.optimizer_D_seg.zero_grad()
        self.backward_D_seg()
        self.optimizer_D_seg.step()

    def gan_visual(self,epoch):
        collections=[]
        for idx, im in enumerate([self.real_A, self.x_ab_ori, self.real_A_mask, self.fake_B_mask, \
                   self.real_gc_A, self.x_gc_ab_ori, self.real_A_gc_mask, self.fake_gc_B_mask,\
                   self.real_B, self.fake_A_ori, self.real_B_mask, self.x_a_GT]):
            if idx % 4 == 2 or idx % 4 == 3: 
                tim = (im[0,0].detach().cpu().numpy()) > 0
                tim = tim.astype(np.uint8)
            else:
                tim= np.clip(((im[0,0].detach().cpu().numpy())+1)*127.5,0,255).astype(np.uint8)
            collections.append(tim)
        for i in range(3):
            for j in range(4):
                plt.subplot(3,4,i*4+j+1)
                plt.imshow(collections[i*4+j],cmap='gray')
                plt.axis('off')
        
        plt.tight_layout()
        e='%03d'%epoch
        plt.savefig(f'{self.opts.name}/i2i_train_visual/{e}_{time.time()}.png',dpi=200)
        plt.close()

    def save(self,  epoch):
        model_name = os.path.join(self.opts.name,'i2i_checkpoints', 'enc_%04d.pt' % (epoch + 1))
        torch.save({'netG_AB': self.netG_AB.state_dict(), 'segnet_ab': self.segnet_ab.state_dict(),
                    'segnet_ba': self.segnet_ba.state_dict(), 'netF_AB': self.netF_AB.state_dict(),
                    'netF_BA': self.netF_BA.state_dict()}, model_name)
