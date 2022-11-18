#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils



def mpjpe_error(batch_pred,batch_gt): 
 
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred, 2, 1))
    
    
def final_mpjpe_error(batch_pred,batch_gt): 
    
    batch_pred=batch_pred[:, -1, :, :].contiguous().view(-1,3)
    batch_gt=batch_gt[:, -1, :, :].contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred, 2, 1))

def avo_loss(batch_pred, batch_gt, epoch, args):
    # pose_pred: [256, 35, 22, 3], pose_gt [256, 35, 22, 3]
    batch_pred = batch_pred.reshape(-1,args.output_n,22,3)
    # [256, 35, 22, 3]
    batch_gt = batch_gt.to(args.device).reshape(-1,args.output_n,22,3)
    # [256, 35, 22, 3]
    batch_gt_var = torch.std(batch_gt.contiguous(),dim=1)
    # [256, 22, 3]
    batch_gt_var = torch.std(batch_gt_var,dim=0).unsqueeze(0).unsqueeze(0) # std on batch
    # [1, 1, 22, 3]
    batch_gt_var = torch.nn.functional.normalize(batch_gt_var, dim=2, p=1).repeat(batch_pred.shape[0], args.output_n, 1, 1)
    # [256, 25, 22, 3]
    errors = batch_gt.contiguous() - batch_pred.contiguous()
    # [256, 25, 22, 3]
    var_error = torch.mul(errors, batch_gt_var)
    # [256, 25, 22, 3]
    if epoch == args.n_epochs:
        kl = 1
    else:
        kl = 1 / (args.n_epochs - epoch)
    loss = torch.mean(torch.norm((errors + (kl * var_error)).view(-1,3),2,1))

    return loss

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def velocity_loss(pose_pred, pose_gt, args):
    # pose_pred: [256, 35, 22, 3], pose_gt [256, 35, 22, 3]
    if pose_pred.shape[1] == args.output_n:
        d_pred = pose_pred.reshape(-1,args.output_n,22,3)
        # [256, 35, 22, 3]
        d_pred = gen_velocity(d_pred)
        # [256, 34, 22, 3]
        d_gt = pose_gt.reshape(-1,args.output_n,22,3)
        # [256, 35, 22, 3]
        d_gt = gen_velocity(d_gt)
        # [256, 34, 22, 3]
    else:
        assert pose_pred.shape[1] == args.seq_len
        d_pred = pose_pred.reshape(-1,args.seq_len,22,3)
        # [256, 35, 22, 3]
        d_pred = gen_velocity(d_pred)
        # [256, 34, 22, 3]
        d_gt = pose_gt.reshape(-1,args.seq_len,22,3)
        # [256, 35, 22, 3]
        d_gt = gen_velocity(d_gt)
        # [256, 34, 22, 3]
    dloss = torch.mean(torch.norm((d_pred - d_gt).reshape(-1,3), 2, 1))
    return dloss


