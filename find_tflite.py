import json
import logging
import os
from itertools import chain
import copy
import argparse
import numpy as np
from EarlyExits.models.efficientnet import EEEfficientNet
import torch
import torch.nn as nn
import onnx

import sys
sys.path.append(os.getcwd())

from train_utils import get_data_loaders, get_optimizer, get_loss, get_lr_scheduler, initialize_seed, train, validate, load_checkpoint, Log
from utils import get_network_search # download CNAS for OFA supernet handling

from EarlyExits.evaluators import sm_eval, binary_eval, standard_eval, ece_score
from EarlyExits.trainer import binary_bernulli_trainer, joint_trainer
from EarlyExits.utils_ee import get_ee_efficientnet, get_intermediate_backbone_cost, get_intermediate_classifiers_cost, get_subnet_folder_by_backbone, get_eenn
import torchvision.models as models

#--trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000
#init_lr=0.01, lr_schedule_type='cosine' weight_decay=4e-5, label_smoothing=0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mobilenetv3', help='name of the model (mobilenetv3, ...)')
    parser.add_argument('--ofa', action='store_true', default=True, help='s')
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    #parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Base learning rate at the start of the training.") #0.1
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--n_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="L2 weight decay.")
    parser.add_argument('--val_split', default=0.0, type=float, help='use validation set')
    parser.add_argument('--optim', type=str, default='SGD', help='algorithm to use for training')
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument("--data_aug", default=True, type=bool, help="True if you want to use data augmentation.")
    parser.add_argument('--save', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for training / testing')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the given dataset')
    parser.add_argument('--supernet_path', type=str, default='./ofa_nets/ofa_mbv3_d234_e346_k357_w1.0', help='file path to supernet weights')
    parser.add_argument('--model_path', type=str, default=None, help='file path to subnet')
    parser.add_argument('--output_path', type=str, default=None, help='file path to save results')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use pretrained weights')
    parser.add_argument('--mmax', type=float, default=1000, help='maximum number of MACS allowed')
    parser.add_argument('--top1min', type=float, default=0.0, help='minimum top1 accuracy allowed')
    parser.add_argument("--use_early_stopping", default=True, type=bool, help="True if you want to use early stopping.")
    parser.add_argument("--early_stopping_tolerance", default=5, type=int, help="Number of epochs to wait before early stopping.")
    parser.add_argument("--resolution", default=32, type=int, help="Image resolution.")
    parser.add_argument("--func_constr", action='store_true', default=False, help='use functional constraints')

    #method: bernulli
    parser.add_argument("--method", type=str, default='bernulli', help="Method to use for training: bernulli or joint")
    parser.add_argument("--fix_last_layer", default=True, action='store_true', help="True if you want to fix the last layer of the backbone.")
    parser.add_argument("--gg_on", default=False, action='store_true', help="True if you want to use the global gate.")
    parser.add_argument("--load_backbone_from_archive", default=False, action='store_true', help="True if you want to use a pre-trained backbone from archive")
    parser.add_argument('--eval_test', action='store_true', default=True, help='evaluate test accuracy')
    parser.add_argument("--backbone_epochs", default=5, type=int, help="Number of epochs to train the backbone.")
    parser.add_argument("--warmup_ee_epochs", default=2, type=int, help="Number of epochs to warmup the EENN")
    parser.add_argument("--ee_epochs", default=0, type=int, help="Number of epochs to train the EENN using the support set")
    parser.add_argument("--priors", default=0.5, type=float, help="Prior probability for the Bernoulli distribution.")
    parser.add_argument("--joint_type", default='losses', type=str, help="Type of joint training: logits, predictions or losses.")
    parser.add_argument("--beta", default=1, type=float, help="Beta parameter for the Bernoulli distribution.")
    parser.add_argument("--sample", default=False, type=bool, help="True if you want to sample from the Bernoulli distribution.")
    #parser.add_argument("--recursive", default=True, type=bool, help="True if you want to use recursive training.") #not used
    parser.add_argument("--normalize_weights", default=True, type=bool, help="True if you want to normalize the weights.")
    parser.add_argument("--prior_mode", default='ones', type=str, help="Mode for the prior: ones or zeros.")
    parser.add_argument("--regularization_loss", default='bce', type=str, help="Loss for the regularization.")
    parser.add_argument("--temperature_scaling", default=True, type=bool, help="True if you want to use temperature scaling.")
    parser.add_argument("--regularization_scaling", default=False, type=bool, help="True if you want to use regularization scaling.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability.")
    parser.add_argument("--support_set", default=False, action='store_true', help="True if you want to use the support set.")
    parser.add_argument("--w_alpha", default=1.0, type=float, help="Weight for the accuracy loss.")
    parser.add_argument("--w_beta", default=1.0, type=float, help="Weight for the MACs loss.")
    parser.add_argument("--w_gamma", default=1.0, type=float, help="Weight for the calibration loss.")
    parser.add_argument("--train_weights", default=False, action='store_true', help="True if you want to train the weights.")
    parser.add_argument("--tune_epsilon", default=False, action='store_true', help="True if you want to tune the epsilon.")

    # Custom Argument
    parser.add_argument("--onnx_filename", default="cnas_onnx", type=str, help="Give the name of onnx file")

    args = parser.parse_args()

    # write down all the arguments to the log.txt file
    logging.info(args)
    print("current iteration: {}".format(args.model_path))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    logging.info('Experiment dir : {}'.format(args.output_path))

    fh = logging.FileHandler(os.path.join(args.output_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = args.device
    use_cuda=False
    if torch.cuda.is_available() and device != 'cpu':
        device = 'cuda:{}'.format(device)
        logging.info("Running on GPU")
        use_cuda=True
    else:
        logging.info("No device found")
        logging.warning("Device not found or CUDA not available.")
    
    device = torch.device(device)
    initialize_seed(42, use_cuda)

    if args.method == 'bernulli':
        get_binaries = True
    else:
        get_binaries = False

    early_stopping = None

    fix_last_layer = False
    if get_binaries:
        fix_last_layer = args.fix_last_layer
    
    if args.dataset=='cifar100':
        n_classes=100
    elif args.dataset=='ImageNet16':
        n_classes=120
    else:
        n_classes=10

    if 'mobilenetv3' in args.model:
        n_subnet = args.output_path.rsplit("_", 1)[1]
        save_path = os.path.join(args.output_path, 'net_{}.stats'.format(n_subnet))

        supernet_path = args.supernet_path
        if args.model_path is not None:
            model_path = args.model_path
        logging.info("Model: %s", args.model)
        
        backbone, res = get_network_search(model=args.model,
                                    subnet_path=args.model_path, 
                                    supernet=args.supernet_path, 
                                    n_classes=n_classes, 
                                    pretrained=args.pretrained,
                                    func_constr=args.func_constr)
    else:
        backbone=models.efficientnet_b0(weights='DEFAULT') #EEEfficientNet()
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # Dropout for regularization
            nn.Linear(1280, n_classes, bias=True)  # Fully connected layer
        )
        save_path = os.path.join(args.output_path, 'net.stats')
        res = args.resolution

    if res is None:
        res = args.resolution

    logging.info(f"DATASET: {args.dataset}")
    logging.info("Resolution: %s", res)
    logging.info("Number of classes: %s", n_classes)
    print("EE epochs: ", args.ee_epochs)

    train_loader, val_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=args.batch_size, threads=args.n_workers, 
                                            val_split=args.val_split, img_size=res, augmentation=True, eval_test=args.eval_test)
    
    if val_loader is not None:
        n_samples=len(val_loader.dataset)
    else:
        val_loader = test_loader
        n_samples=len(test_loader.dataset)

    print("Train samples: ", len(train_loader.dataset))
    print("Val samples: ", len(val_loader.dataset))

    train_log = Log(log_each=10)

    #parameters = chain(backbone.parameters(), classifiers.parameters())

    optimizer = get_optimizer(backbone.parameters(), args.optim, args.learning_rate, args.momentum, args.weight_decay)

    criterion = get_loss('ce')
    
    scheduler = get_lr_scheduler(optimizer, 'cosine', epochs=args.backbone_epochs)

    if (os.path.exists(os.path.join(args.output_path,'backbone.pth'))):

        backbone, optimizer = load_checkpoint(backbone, optimizer, device, os.path.join(args.output_path,'backbone.pth'))
        logging.info("Loaded checkpoint")
        top1 = validate(val_loader, backbone, device, print_freq=100)/100

    else:

        if args.backbone_epochs > 0:
            logging.info("Start training...")
            top1, backbone, optimizer = train(train_loader, val_loader, args.backbone_epochs, backbone, device, optimizer, criterion, scheduler, train_log, ckpt_path=os.path.join(args.output_path,'backbone.pth'))
            logging.info("Training finished")
    
    if args.backbone_epochs == 0:
        top1 = validate(val_loader, backbone, device, print_freq=100)/100
    logging.info(f"VAL ACCURACY BACKBONE: {np.round(top1*100,2)}")
    if args.eval_test:
        top1_test = validate(test_loader, backbone, device, print_freq=100)
        logging.info(f"TEST ACCURACY BACKBONE: {top1_test}")
    
    results={}
    results['backbone_top1'] = np.round((1-top1)*100,2)

    #Create the EENN on top of the trained backbone

    if 'mobilenetv3' in args.model:
        backbone, classifiers, epsilon = get_eenn(subnet=backbone, subnet_path=args.model_path, res=res, n_classes=n_classes, get_binaries=get_binaries)
    else:
        backbone, classifiers, epsilon = get_ee_efficientnet(model=backbone, img_size=res, n_classes=n_classes, get_binaries=get_binaries)

    # NOTE: args.model_path = results/search_path/iter_0/net_0/net_0.subnet. This changes for each subnet #

    print("--------------------------------")
    param_size = 0
    buffer_size = 0
    for param in backbone.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in backbone.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Size: {:.3f} MB'.format(size_all_mb))
    logging.info('Model Size: {:.3f} MB'.format(size_all_mb))

    # NOTE: ONNX gets created here
    torch_input = torch.randn(1, 3, res, res).to(device) # device = cuda:{}. It has to run on GPU & 4D
    filename = args.model_path.replace('/','_').replace('.','_')
    torch.onnx.export(backbone, torch_input, '{}.onnx'.format(args.onnx_filename), opset_version=11)  # create onnx file

    print("{}.onnx created".format(args.onnx_filename))

    print("------------------------------\n")
