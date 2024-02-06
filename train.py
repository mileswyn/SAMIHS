import os
import sys
import torch
import numpy as np
import random
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group as ipg
import torch.multiprocessing as mp

from utils import I2IDataset_T2C, I2IDataset_C2T, create_dirs, calc_loss
from trainer import Solver

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

def main_train(rank, worldsize):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='/path/to/save/')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2)
    opts = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    check_manual_seed(opts.seed)
    create_dirs(opts.name)
    ipg(backend='gloo', world_size=worldsize, rank=rank)
    train_sampler = DistributedSampler(I2IDataset_T2C(train=True), shuffle=True) # I2IDataset_C2T
    train_loader = DataLoader(dataset=I2IDataset_T2C(train=True),
                              batch_size=opts.bs,
                              num_workers=0,
                              pin_memory=True,
                              sampler=train_sampler)
    valid_sampler = DistributedSampler(I2IDataset_T2C(train=False), shuffle=False)
    valid_loader = DataLoader(dataset=I2IDataset_T2C(train=False),
                              batch_size=1,
                              num_workers=0,
                              pin_memory=True,
                              sampler=valid_sampler)
    trainer = Solver(opts, cuda=rank)
    trainer.cuda(rank)
    iteration=0

    trainer.netG_AB = DDP(trainer.netG_AB, device_ids=[rank], output_device=rank)
    trainer.netG_BA = DDP(trainer.netG_BA, device_ids=[rank], output_device=rank)
    trainer.netD_B = DDP(trainer.netD_B, device_ids=[rank], output_device=rank)
    trainer.netD_A = DDP(trainer.netD_A, device_ids=[rank], output_device=rank)
    trainer.netD_gc_B = DDP(trainer.netD_gc_B, device_ids=[rank], output_device=rank)
    trainer.netD_gc_A = DDP(trainer.netD_gc_A, device_ids=[rank], output_device=rank)

    for epoch in range(200):
        trainer.train()
        for idx, train_data in enumerate(train_loader):
            for k in train_data.keys():
                train_data[k] = train_data[k].cuda(rank).detach()
            if epoch == 0 and idx == 0:
                trainer.initialize_NCE(train_data['A_img'], train_data['B_img'], train_data['A_img_GT'])
            trainer.gan_forward(train_data['A_img'], train_data['B_img'], train_data['A_img_GT'], epoch)
            trainer.writerEpoch += 1
            if iteration%100==0:
                trainer.gan_visual(epoch)
            sys.stdout.write(f'\r Epoch {epoch}, Iter {iteration}')
            iteration+=1
        with torch.no_grad():
            trainer.eval()
            val_losses = []
            for val_data in valid_loader:
                for k in val_data.keys():
                    val_data[k] = val_data[k].cuda().detach()
                B, B_GT = val_data['B_img'], val_data['B_img_GT']
                pred_mask_b = trainer.test_seg(B)
                loss = calc_loss(pred_mask_b, B_GT)
                val_losses.append(loss.detach().cpu().numpy())
            trainer.tensorboardWriter.add_scalar('validation_seg_loss', np.mean(val_losses), epoch)
        trainer.update_lr()
        trainer.seg_ab_schedule.step()
        trainer.seg_ba_schedule.step()
        if rank==0:
            trainer.save(epoch)

def main():
    # TODO
    mp.spawn(main_train, args=(1, ), nprocs=1, join=True)

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    # TODO
    os.environ['MASTER_PORT'] = '15382'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # print(os.environ)
    main()