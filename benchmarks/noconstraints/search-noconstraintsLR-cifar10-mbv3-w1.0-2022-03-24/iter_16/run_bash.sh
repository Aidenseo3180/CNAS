#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_0.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_0.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_1.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_1.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_2.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_2.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_3.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_3.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_4.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_4.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_5.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_5.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_6.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_6.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_7.subnet --data ../data/cifar10 --dataset cifar10 --n_classes 10 --supernet ../data/ofa_mbv3_d234_e346_k357_w1.0 --pretrained --pmax 2.2 --fmax 7.0 --amax 0.3 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/search-cifar10-mbv3-w1.0-2022-03-24/iter_16/net_7.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 5000 --reset_running_statistics &
wait
