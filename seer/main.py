from train import *
import torchvision
from data import *
from PIL import Image
import time
from model import *
import torch
import distr

args = get_args()
args.prop_mode = 'max'
args.batch_size_train = '63_1_2'
args.batch_size_test = '63_1'
args.acc_grad = 10
args.learning_rate = 1e-4
args.data_path = '../data'
args.res_path ='../models/cifar10'
args.attack_cfg ='modern'
args.print_interval = 30
args.num_epochs = 200
args.epoch_steps = 1000
args.test_interval = 50
args.big_test_interval = 10000
args.act = 'ReLU'
args.public_labels = True
args.par_sel_size = 8400
args.par_sel_frac = 0.001
args.mid_rep_frac = 1.0
torch.manual_seed(args.rng_seed)
torch.set_default_dtype(torch.float32)
normal=torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
batch_norm=torch.nn.BatchNorm1d(1, eps=1e-05, affine=False, track_running_stats=False, device=args.device, dtype=None)


# CIFAR/TinyImageNet first layer
# https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/backbones/resnet_mmcls.py#L716
assert args.dataset in ['Cifar10', 'Cifar100', 'TinyImageNet', 'TinyImageNet_rsz', 'Cifar10_2', 'Cifar10_1'] or args.dataset.startswith( 'Cifar10_C' )
public_model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], args.act, num_classes=args.num_classes).to(args.device)

par_sel=ParamSelector(public_model,args.par_sel_size,args.par_sel_frac,sparse_grad=False,seed=98)
grad_ex=GradientExtractor(public_model,par_sel).to(args.device)
disaggregator, reconstructor = get_decoder(args, grad_ex)

optimizer = optim.Adam( list(public_model.parameters())+list(disaggregator.parameters())+list(reconstructor.parameters()), lr=args.learning_rate)

if args.dataset.startswith('Cifar10_C'):
    C, sev = args.dataset[10:].rsplit( '_', 1 )
    sev = int(sev)
    trainset, testset = globals()[f'datasets_Cifar10_C'](C,sev)
else:
    trainset, testset = globals()[f'datasets_{args.dataset}']()
if (args.prop_mode == 'thresh') and (args.thresh is None) and (args.task!="secagge2e"):
    args.thresh=distr.compute_thresh(dataset=trainset,prop=args.prop,batch_size=(args.batch_size_train[0] + args.batch_size_train[1]),num_clients=args.num_clients,num_samples=args.est_thr,bn=batch_norm)
    print(f"THRESHOLD for (trainset={args.dataset},prop={args.prop},batch_size={(args.batch_size_train[0] + args.batch_size_train[1])},num_clients={args.num_clients}): ",args.thresh)


if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
else:
    checkpoint = None
modules=[public_model, grad_ex, disaggregator, reconstructor]
if args.task == 'train': 
    train(args, modules, optimizer, trainset, testset, batch_norm,checkpoint=checkpoint)
elif args.task == 'test':
    tests(args, modules, trainset, testset, batch_norm,checkpoint=checkpoint)
elif args.task == 'end2end':
    test_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'end2end_contrast':
    test_end2end_fix_contrast(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'secaggr':
    test_sec_aggr(args, modules, trainset, testset, checkpoint=checkpoint,metr=True)
elif args.task == 'secagge2e':
    test_sec_aggr_end2end(args, modules, trainset, testset, batch_norm,checkpoint=checkpoint,metr=True)
elif args.task == 'baseline':
    baseline_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
elif args.task == 'tests':
    test_sec_aggr(args, modules, trainset, testset, batch_norm,checkpoint=checkpoint)
    test_sec_aggr_end2end(args, modules, trainset, testset, checkpoint=checkpoint)
else:
    tests(args, modules, loader_train, loader_test,batch_norm,checkpoint=checkpoint, vis_res=True)
