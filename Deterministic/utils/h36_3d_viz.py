#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utils.loss_funcs import final_mpjpe_error
from utils.data_utils import define_actions, joint_to_ignore, joint_equal
import os


def create_pose(ax,plots,vals,pred=True,update=False,prediction=False):

            
    
    # h36m 32 joints(full)
    connect = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (6, 7), (7, 8), (8, 9), (9, 10),
            (0, 1), (0, 6),
            (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (24, 25), (24, 17),
            (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  


# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred and prediction:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:

            if i ==0 and not pred:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    #         if i == 0 and pred and prediction:
    #             plots[i][0].set_label(['GT' if not pred else 'Pred'])
        
    # if i == 0 and pred and prediction:
    #     ax.legend(loc='lower left')
    
    return plots
   # ax.legend(loc='lower left')


# In[11]:


def update(num,data_gt,data_pred,plots_gt,plots_pred,fig,ax,input_n):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True,prediction=num>=input_n)
    plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True,prediction=num>=input_n)
    
    

    
    
    r = 0.75
    xroot, zroot, yroot = data_gt[0][0,0], data_gt[0][0,1], data_gt[0][0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred
    


# In[12]:


def visualize(modello, args):
    
    input_n = args.input_n
    output_n = args.output_n
    visualize_from = args.visualize_from
    device = args.device
    n_viz = args.n_viz
    actions = args.viz_actions
    global_translation = args.global_translation
    model_name = args.name
    
    if not global_translation:
        from utils import h36motion3d as datasets
    else:
        args.dim_used.sort()
        from utils import h36motion3dab as datasets
    
    os.makedirs('./gifs/'+model_name, exist_ok=True)

    for action in actions:
    
        if visualize_from=='train':
            loader=datasets.Datasets(args, split=0,actions=[action])
        elif visualize_from=='validation':
            loader=datasets.Datasets(args, split=1,actions=[action])
        elif visualize_from=='test':
            loader=datasets.Datasets(args, split=2,actions=[action])
            
      # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            
            
        loader = DataLoader(
        loader,
        batch_size=1,
        shuffle = False, # for comparable visualizations with other models
        num_workers=0)       
        
            
    
        for cnt,batch in enumerate(loader): 
            batch = batch.to(device) 
            
            all_joints_seq=batch.clone()[:, 0:input_n+output_n,:]
            
            sequences_train=batch[:, 0:input_n, args.dim_used].view(-1,input_n,len(args.dim_used)//3,3).permute(0,3,1,2)
            sequences_gt=batch[:, 0:input_n+output_n, :]
            
            sequences_predict, sequences_predict_all=modello(sequences_train)
            sequences_predict=sequences_predict.permute(0,1,3,2).contiguous().view(-1,output_n,len(args.dim_used))
            
            all_joints_seq[:,input_n:input_n+output_n,args.dim_used] = sequences_predict
            
            all_joints_seq[:,input_n:input_n+output_n,index_to_ignore] = all_joints_seq[:,input_n:input_n+output_n,index_to_equal]
            
            
            all_joints_seq=all_joints_seq.view(-1,input_n+output_n,32,3)
            
            sequences_gt=sequences_gt.view(-1,input_n+output_n,32,3)
            
            loss=final_mpjpe_error(all_joints_seq,sequences_gt)# # both must have format (batch,T,V,C)
    
            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000
    
    
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.view_init(elev=20, azim=-40)
            vals = np.zeros((32, 3)) # or joints_to_consider
            gt_plots=[]
            pred_plots=[]
    
            gt_plots=create_pose(ax,gt_plots,vals,pred=False,update=False)
            pred_plots=create_pose(ax,pred_plots,vals,pred=True,update=False)
    
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(loc='lower left')
    
    
    
            ax.set_xlim3d([-1, 1.5])
            ax.set_xlabel('X')
    
            ax.set_ylim3d([-1, 1.5])
            ax.set_ylabel('Y')
    
            ax.set_zlim3d([0.0, 1.5])
            ax.set_zlabel('Z')
            ax.set_title('loss in mm is: '+str(round(loss.item(),4))+' for action : '+str(action)+' for '+str(output_n)+' frames')
    
            line_anim = animation.FuncAnimation(fig, update, input_n + output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots,
                                                                       fig,ax,input_n),interval=70, blit=False)
            plt.show()
            
            line_anim.save('./gifs/'+model_name+'/human_viz_{}_{}.gif'.format(str(action), cnt),writer='pillow')
    
            
            if cnt==n_viz-1:
                break

def my_visualize(modello, args):
    
    input_n = args.input_n
    output_n = args.output_n
    skip_rate = args.skip_rate
    n_viz = args.n_viz
    device = args.device
    if args.viz_actions == 'all':
        args.viz_actions = args.datasets.h36m.actions
    visualize_from = args.visualize_from
    modello = modello.to(device)
    actions = args.viz_actions
    
    if not args.global_translation:
        from utils import h36motion3d as datasets
    else:
        args.dim_used.sort()
        from utils import h36motion3dab as datasets
    
    for action in actions:
    
        if visualize_from=='train':
            loader=datasets.Datasets(args, split=0,actions=[action], verbose=False)
        elif visualize_from=='val':
            loader=datasets.Datasets(args, split=1,actions=[action], verbose=False)
        elif visualize_from=='test':
            loader=datasets.Datasets(args, split=2,actions=[action], verbose=False)
            
      # joints at same loc
        index_to_ignore = np.concatenate((joint_to_ignore * args.data_dim, joint_to_ignore * args.data_dim + 1, joint_to_ignore * args.data_dim + 2))
        index_to_equal = np.concatenate((joint_equal * args.data_dim, joint_equal * args.data_dim + 1, joint_equal * args.data_dim + 2))
            
            
        loader = DataLoader(loader, batch_size=1, shuffle = False, num_workers=args.num_workers)       
        
            
    
        for cnt,batch in enumerate(loader): 
            batch = batch.to(device) 
            
            all_joints_seq=batch.clone()[:, 0:input_n+output_n,:]
            
            sequences_train=batch[:, 0:input_n, args.dim_used].view(-1,input_n,len(args.dim_used)//3,3).permute(0,3,1,2)
            sequences_gt=batch[:, 0:input_n+output_n, :]
            
            sequences_predict, _ = modello(sequences_train)
            sequences_predict=sequences_predict.permute(0,1,3,2).contiguous().view(-1,output_n,len(args.dim_used))
            
            all_joints_seq[:,input_n:input_n+output_n, args.dim_used] = sequences_predict
            
            all_joints_seq[:,input_n:input_n+output_n,index_to_ignore] = all_joints_seq[:,input_n:input_n+output_n,index_to_equal]
            
            
            all_joints_seq=all_joints_seq.view(-1,input_n+output_n,32,3)
            
            sequences_gt=sequences_gt.view(-1,input_n+output_n,32,3)
               
            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000
    
    
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.view_init(elev=20, azim=-40)
            vals = np.zeros((32, 3)) # or joints_to_consider
            gt_plots=[]
            pred_plots=[]
    
            gt_plots=create_pose(ax,gt_plots,vals,pred=False,update=False)
            pred_plots=create_pose(ax,pred_plots,vals,pred=True,update=False)
    
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(loc='lower left')
    
    
    
            ax.set_xlim3d([-1, 1.5])
            ax.set_xlabel('X')
    
            ax.set_ylim3d([-1, 1.5])
            ax.set_ylabel('Y')
    
            ax.set_zlim3d([0.0, 1.5])
            ax.set_zlabel('Z')
    
            line_anim = animation.FuncAnimation(fig, update, input_n + output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots,
                                                                       fig,ax,input_n),interval=70, blit=False)
            
            line_anim.save(os.path.join(args.viz_path, action, 'human_viz_{}.gif'.format(cnt + 1)), writer='pillow')
    
            plt.close()
            if cnt==n_viz-1:
                break