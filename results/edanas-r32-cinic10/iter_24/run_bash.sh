#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_1.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_1.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_2.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_2.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_3.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_3.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_4.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_4.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_5.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_5.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_6.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_6.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
CUDA_VISIBLE_DEVICES=0 python evaluator.py --subnet ../results/edanas-r32-cinic10/iter_24/net_7.subnet --data ../../datasets/cinic10 --dataset cinic10 --n_classes 10 --supernet ./ofa_nets/ofa_eembv3 --pretrained --pmax 2.0 --fmax 100 --amax 5.0 --wp 1 --wf 0.025 --wa 1 --penalty 10000000000 --save ../results/edanas-r32-cinic10/iter_24/net_7.stats --trn_batch_size 128 --vld_batch_size 200 --num_workers 4 --n_epochs 5 --resolution 224 --valid_size 9000 --reset_running_statistics &
wait
