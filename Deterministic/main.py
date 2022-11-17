import os
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.data_utils import define_actions, joint_equal, joint_to_ignore
from utils.loss_funcs import *
from utils.h36_3d_viz import visualize, my_visualize
from utils.parser import args
from tqdm import tqdm
from model import Model
from utils.utils import get_dataset_and_loader, line_prepender, save_results

device = args.device
print('Using device: %s'%device)

model = Model(3,args.input_n,
                          args.output_n,args.st_gcnn_dropout,args.dim_used,args.n_pre,args.version).to(device)
print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
model_name='h36_3d_'+str(args.input_n)+'_'+str(args.output_n)+'_'+str(args.skip_rate)+'_'+str(args.n_pre)+'_'+args.version+'_ckpt_'+['local','global'][args.global_translation]

def train():
    # dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=0)
    # print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    data_loader = get_dataset_and_loader(args=args, split='train', actions=args.actions)
    # vald_dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=1)
    # print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    # vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # vald_loader = get_dataset_and_loader(args=args, split='val', actions=args.actions)

    optimizer=optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    if args.wandb:
      wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=args.name,
        config=args
      )
      line_prepender(args.config_path, wandb.run.get_url())
      
    train_loss = []
    val_loss = []

    for epoch in range(args.n_epochs):
      running_loss=0
      n=0
      model.train()
      for iteration, batch in tqdm(enumerate(data_loader)): 
          batch=batch.to(device)
          # [256, 35, 96]
          batch_dim=batch.shape[0]
          n+=batch_dim
          
          sequences_train=batch[:, 0:args.input_n, args.dim_used].view(-1, args.input_n, args.num_joints, args.data_dim).permute(0,3,1,2)
          # [256, 3, 10, 22]
          sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, args.dim_used].view(-1, args.output_n, args.num_joints, args.data_dim)
          # [256, 25, 22, 3]
          sequences_gt_all=batch[:, :args.input_n+args.output_n, args.dim_used].view(-1, args.input_n+args.output_n, args.num_joints, args.data_dim)
          # [256, 35, 22, 3]
          
          optimizer.zero_grad() 

          sequences_predict, sequences_predict_all=model(sequences_train)
          # [256, 25, 3, 22], [256, 35, 3, 22]
          sequences_predict = sequences_predict.permute(0,1,3,2)
          # [256, 25, 22, 3]
          sequences_predict_all = sequences_predict_all.permute(0,1,3,2)
          # [256, 35, 22, 3]
          loss = torch.tensor(0).float().to(args.device)
          log_dict = {}
          if args.pred_loss:
            prediction_loss = mpjpe_error(sequences_predict,sequences_gt)
            loss += args.coeff_pred * prediction_loss
            log_dict["train/prediction"] = prediction_loss.item()
          if args.reco_loss:
            fullmotion_loss = mpjpe_error(sequences_predict_all,sequences_gt_all)
            loss += args.coeff_reco * fullmotion_loss
            log_dict["train/reconstruction"] = fullmotion_loss.item()
          if args.vel_loss:
            vel_loss = velocity_loss(sequences_predict, sequences_gt, args)
            loss += args.coeff_vel * vel_loss
            log_dict["train/velocity"] = vel_loss.item()
          if args.avo_loss:
            avo_loss_ = avo_loss(sequences_predict, sequences_gt)
            loss += args.coeff_avo * avo_loss_
            log_dict["train/avo"] = avo_loss_.item()
          log_dict["train/total_loss"] = loss.item()

          if iteration % args.log_every == 0:
            print('[%d, %5d]  mpjpe loss: %.3f' %(epoch + 1, iteration + 1, loss.item())) 
            if args.wandb:
              wandb.log(log_dict)
          
          loss.backward()
          if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
          optimizer.step()
          running_loss += loss*batch_dim
      
      print('[%d]  mpjpe loss: %.3f' %(epoch + 1, running_loss.detach().cpu()/n)) 
      epoch_results = {}
      
      train_loss.append(running_loss.detach().cpu()/n)  
      epoch_results['train/epoch_loss'] = train_loss[-1].item()
      
      if args.use_scheduler:
        scheduler.step()
      
      
      if (epoch + 1) % args.val_every == 0:
        vald_error_on_test_frame = test(split="val")
        val_loss.append(vald_error_on_test_frame)
        epoch_results['train/validation_MPJPE'] = vald_error_on_test_frame
        print('[%d, all]  validation loss: %.3f' %(epoch + 1, val_loss[-1]))
      
      if (epoch + 1) % args.test_every == 0:
        test_error_on_test_frame = test(split="test", epoch=epoch, wandblog=True)
        epoch_results['train/test_MPJPE'] = test_error_on_test_frame
      
      if args.wandb:
        wandb.log(epoch_results)

      print('----saving last model-----')
      torch.save(model.state_dict(), args.ckpt_path)
      
      if val_loss[-1] == min(val_loss):
        print('----saving best model-----')
        torch.save(model.state_dict(), args.best_path)
      
      elif (epoch+1) % args.save_every == 0:
        print('----saving model-----')
        torch.save(model.state_dict(), args.epoch_path.format(str(epoch+1)))

        # plt.figure(1)
        # plt.plot(train_loss, 'r', label='Train loss')
        # plt.plot(val_loss, 'g', label='Val loss')
        # plt.legend()
        # plt.show()

def test(split="test", wandblog=False, tablelog=False, epoch=0):
  assert args.output_n >= args.test_output_n
  # if args.load_checkpoint:
  #   model.load_state_dict(torch.load(args.best_path))
  model.eval()
  n_batches = 0 # number of batches for all the sequences
  actions = args.actions
  # joints at same loc
  index_to_ignore = np.concatenate((joint_to_ignore * args.data_dim, joint_to_ignore * args.data_dim + 1, joint_to_ignore * args.data_dim + 2))
  index_to_equal = np.concatenate((joint_equal * args.data_dim, joint_equal * args.data_dim + 1, joint_equal * args.data_dim + 2))
  MPJPE_errors = np.zeros(len(args.eval_frames))
  errors_per_action = [] 
  idx_eval = args.test_output_n
  if tablelog:
    res_dict = {}
  
  for action in actions:
    n=0
    action_errors = np.zeros(len(args.eval_frames))
    test_loader = get_dataset_and_loader(args, split=split, actions=[action], verbose=False)
    
    for iteration, batch in enumerate(test_loader):
      with torch.no_grad():
        
        batch=batch.to(device)
        # [256, 35, 96]
        batch_dim=batch.shape[0]
        n+=batch_dim
        
        all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]
        # [256, 25, 96]

        sequences_train=batch[:, 0:args.input_n, args.dim_used].view(-1, args.input_n, args.num_joints, args.data_dim).permute(0,3,1,2)
        # [256, 3, 10, 22]
        sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]
        # [256, 25, 96]
        
        sequences_predict, _ =model(sequences_train)
        # [256, 25, 3, 22]
        sequences_predict = sequences_predict.permute(0,1,3,2).contiguous().view(-1,args.output_n, args.num_joints * args.data_dim)
  	    # [256, 25, 66]
        
        all_joints_seq[:,:,args.dim_used] = sequences_predict
  	    # [256, 25, 96]

        all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]
        # [256, 25, 96]
        
        for k, frame in enumerate(args.eval_frames):
          if args.metric == 'literature':
            prediction = all_joints_seq.view(-1,args.output_n,32,3)[:, :frame, :, :]
            gt = sequences_gt.view(-1,args.output_n,32,3)[:,:frame, :, :]
            loss=final_mpjpe_error(prediction, gt)
            action_errors[k] = loss.item() * batch_dim
            MPJPE_errors[k] += loss.item() * batch_dim
          elif args.metric == 'ours':
            prediction = all_joints_seq[:,:, args.dim_used].view(-1,args.output_n,22,3)[:,:frame, :, :]
            gt = sequences_gt[:,:, args.dim_used].view(-1,args.output_n,22,3)[:,:frame, :, :]
            loss=final_mpjpe_error(prediction, gt)
            action_errors[k] = loss.item() * batch_dim
            MPJPE_errors[k] += loss.item() * batch_dim
        
        # if args.metric == 'literature': 
        #   loss=final_mpjpe_error(all_joints_seq.view(-1,args.output_n, 32, 3)[:,:args.test_output_n,:,:],sequences_gt.view(-1,args.output_n,32,3)[:,:args.test_output_n,:,:])
        # elif args.metric == 'ours':
        #   loss=final_mpjpe_error(all_joints_seq[:,:, args.dim_used].view(-1,args.output_n,22,3)[:,:args.test_output_n,:,:],sequences_gt[:,:, args.dim_used].view(-1,args.output_n,22,3)[:,:args.test_output_n,:,:])
      
        # running_loss+=loss*batch_dim
        # accum_loss+=loss*batch_dim
    
    action_errors /= n
    
    print('loss at test subject for action : ' + str(action) + ' is: ' + str(round(action_errors[idx_eval], 2)))
    errors_per_action.append(action_errors[idx_eval])
    if (args.wandb) and (wandblog):
        wandb.log({"actions/{}@{}".format(action, str(args.eval_frames[idx_eval])): action_errors[idx_eval]})
    if tablelog:
        res_dict[action] = action_errors[idx_eval]
    n_batches += n
  
  MPJPE_errors /= n_batches 
   
  print('mean on all samples (mm) is:', round((MPJPE_errors[idx_eval]).item(), 2))
  print('average on test actions (mm) is: ', round(np.mean(errors_per_action), 2))      
  if (args.wandb) and (wandblog):
      for frame, mpjpe_puntual in zip(args.eval_frames, MPJPE_errors):
          wandb.log({"frames/MPJPE@{}".format(str(frame)): mpjpe_puntual})
  if tablelog:
      res_dict["average"] = MPJPE_errors[idx_eval]
      save_results(res_dict, args)
  
  if (args.wandb) and (wandblog):
      my_visualize(model, args)
      for action in args.viz_actions:
          gifpath = os.path.join(args.viz_path, action, 'human_viz_1.gif')
          wandb.log({"gif/{}/e{}".format(action, epoch): wandb.Video(gifpath, fps=4, format="gif")})

  return MPJPE_errors[idx_eval]

  

if __name__ == '__main__':

    if args.mode == 'train':
      train()
      test(tablelog=True)
    elif args.mode == 'test':
      test(tablelog=True)
      my_visualize(model, args)
    elif args.mode=='viz':
       model.load_state_dict(args.best_path)
       model.eval()
       visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider,args.global_translation,model_name)



