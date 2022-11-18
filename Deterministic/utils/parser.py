import argparse
from utils.utils import set_name, set_paths, write_config
from utils.data_utils import dim_used
parser = argparse.ArgumentParser(description='Arguments for running the scripts')


#ARGS FOR THE RUN

parser.add_argument('--name', type=str, help='name of the experiment')
parser.add_argument('--debug', action='store_true', help='Debug session')
parser.add_argument('--seed', type=int, default=304, help= 'seed for reproducibility')
parser.add_argument('--device', type=str, default="cuda:1", choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], help='device of the experiment')



# ARGS FOR PATHs

parser.add_argument('--data_dir', type=str, default='./Deterministic/data', help='path to the unziped dataset directories(H36m)')
parser.add_argument('--root_dir', type=str, default='./checkpoints', help='path to the experiments folder')
parser.add_argument('--table_path', type=str, default='./checkpoints/results.xlsx', help= 'directory with the models checkpoints ')


#ARGS FOR LOADING THE DATASET

parser.add_argument('--dataset_name', type=str, default='h36m', choices=['h36m'], help='name of the dataset')
parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
parser.add_argument('--output_n', type=int, default=10, help="number of model's output frames")
parser.add_argument('--n_pre', type=int, default=35, help="number of dct coefficients")
parser.add_argument('--skip_rate', type=int, default=2, help='rate of frames to skip,defaults=2 for H36M')
parser.add_argument('--global_translation', action='store_true',help='predicting global translation for H36M')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for loading the dataset')


#ARGS FOR THE MODEL

parser.add_argument('--st_gcnn_dropout', type=float, default=.1, help= 'st-gcnn dropout')


#ARGS FOR THE TRAINING

parser.add_argument('--mode', type=str, default='train', choices=['train','test','viz'], help= 'Choose to train,test or visualize from the model.Either train,test or viz')
parser.add_argument('--n_epochs', type=int, default=50, help= 'number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help= 'batch size')
parser.add_argument('--clip_grad',type=float, default=None, help= 'select max norm to clip gradients')
parser.add_argument('--version', type=str, default='long', help= 'model version (long or short)')
parser.add_argument('--save_every', type=int, default=25 ,help='Epochs interval for saving ckpts [0=only last]')
parser.add_argument('--val_every', type=int, default=1 , help='Epochs interval for validation [0=never]')
parser.add_argument('--test_every', type=int, default=1 , help='Epochs interval for validation [0=never]')
parser.add_argument('--log_every', type=int, default=200 , help='Iterations interval for logging [0=never]')

#ARGS FOR LOSSES
parser.add_argument('--vel_loss', type=bool, default=False, help='Use velocity loss')
parser.add_argument('--coeff_vel', type=bool, default=1, help='Velocity loss coefficient')
parser.add_argument('--pred_loss', type=bool, default=True, help='Use prediction loss')
parser.add_argument('--coeff_pred', type=bool, default=1, help='Prediction loss coefficient')
parser.add_argument('--reco_loss', type=bool, default=True, help='Use reconstruction loss')
parser.add_argument('--coeff_reco', type=bool, default=0.5, help='Reconstruction loss coefficient')
parser.add_argument('--avo_loss', type=bool, default=False, help='Use A^2 loss')
parser.add_argument('--coeff_avo', type=bool, default=0.6, help='A^2 loss coefficient')


#ARGS FOR WANDB

parser.add_argument('--wandb', type=bool, default=True, help='Use wandb')
parser.add_argument('--entity', type=str, default='pinlab-sapienza', help= 'WandB entity name')
parser.add_argument('--project',type=str, default='STARS', help= 'WandB project name')
parser.add_argument('--group', type=str, default='STARS', help= 'WandB group name')


#ARGS FOR THE OPTIMIZER

parser.add_argument('--optimizer', type=str, default='adam', choices=['adam'], help='(lowercase) name of the opimizer')
parser.add_argument('--lr',type=int, default=1e-02, help= 'Learning rate of the optimizer')
parser.add_argument('--weight_decay', type=int, default=1e-05, help= 'Constraint on model weights')


#ARGS FOR THE SCHEDULER

parser.add_argument('--use_scheduler', type=bool, default=True, help= 'use MultiStepLR scheduler')
parser.add_argument('--scheduler', type=str, default='multisteplr', choices=['multisteplr'], help= 'type of scheduler')
parser.add_argument('--gamma', type=float, default=0.1, help= 'gamma correction to the learning rate, after reaching the milestone epochs')
parser.add_argument('--milestones', type=list, default=[15,25,35,40], help= 'the epochs after which the learning rate is adjusted by gamma')


#ARGS FOR EVALUATION

parser.add_argument('--metric', type=str, default='ours', choices=['ours', 'literature'], help= 'choose if use only dim_used for evaluation (ours) [ours.traditional]')
parser.add_argument('--batch_size_test', type=int, default=256, help= 'batch size for the test set')
parser.add_argument('--test_output_n', type=int, default=7, help="number of model's test frames")
parser.add_argument('--eval_frames', default=[2, 4, 8, 10, 14, 18, 22, 25], help='frames on which to evaluate')

#FLAGS FOR THE VISUALIZATION

parser.add_argument('--visualize_from', type=str, default='test', choices =['train','val','test'], help= 'choose data split to visualize from(train-val-test)')
parser.add_argument('--viz_actions', default=['walking', 'posing', 'walkingdog', 'smoking'], help= 'Actions to visualize.Choose either all or a list of actions')
parser.add_argument('--n_viz', type=int, default='2', help= 'Numbers of sequences to visaluze for each action')




args = parser.parse_args()

args.seq_len = args.input_n + args.output_n

if args.dataset_name == 'h36m':
    args.num_joints = 22
    args.num_tot_joints = 32
    args.data_dim = 3
    args.actions = [
        'walking',
        'eating',
        'smoking',
        'discussion',
        'directions',
        'greeting',
        'phoning',
        'posing',
        'purchases',
        'sitting',
        'sittingdown',
        'takingphoto',
        'waiting',
        'walkingdog',
        'walkingtogether',
    ]
    if not args.global_translation:
        dim_used = dim_used[12:]
    args.dim_used = dim_used

if args.mode == 'test' or args.mode == 'viz':
    args.load_checkpoint = True
    args.wandb = False
    # args.debug = True
else: 
    args.load_checkpoint = False

if not args.debug:
    args = set_name(args)
    args = set_paths(args)
    write_config(args)
else:
    args.wandb = False

for k, v in args.__dict__.items():
    print(f"{k}: {v}")