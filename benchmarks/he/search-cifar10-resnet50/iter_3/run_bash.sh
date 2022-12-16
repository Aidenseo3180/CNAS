#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_0.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_0.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_1.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_1.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_2.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_2.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_3.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_3.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_4.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_4.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_5.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_5.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_6.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_6.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_7.subnet --data ../data/cifar10 --dataset cifar10HE --n_classes 10 --supernet ../data/ofa_resnet50_he_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0 --pmax 0.5 --fmax 150.0 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-resnet50-2022-04-06/iter_3/net_7.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
