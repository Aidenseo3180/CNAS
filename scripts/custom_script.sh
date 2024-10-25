# minimum requirements to run cnas with multiexit --> create a subnet & inserted in results/ folder

python cnas.py --sec_obj tiny_ml \
              --n_gpus 1 --gpu 1 --n_workers 4 \
              --data datasets/cifar10 --dataset cifar10 \
              --first_predictor as --sec_predictor as \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --save results/search_path --iterations 0 \
              --search_space cbnmobilenetv3 --trainer_type multi_exits \
              --method bernulli --support_set --tune_epsilon\
              --val_split 0.1 \
              --n_epochs 0 --warmup_ee_epochs 0 --ee_epochs 2 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 \
              --mmax 2.7 --top1min 0.65 \
              --lr 32 --ur 32 --rstep 4 \
              --n_doe 10 

# --iteration: number of times you want to run the entire thing
#       - So if iteration = 2 & n_doe = 10 --> entire thing runs (iteration+1)*n_doe
# --n_doe: number of subnets(EENN) it generates per iteration (net_# number comes from idx of n_doe)
#       - Starts from 0. So if n_doe = 10 --> it runs from 0 to 9 (subnets)
#       - This can't be 0 (so has to be at least 1)
# --ee_epochs: turns on support set, a set that helps your EE to train better (really unnecessary)
# --n_epochs: trains the entire epoch 'n' times (ex. if n_epoch is 2, then it trains the model twice --> increase the accuracy significantly)
#       - this shows the graph --> shows you how accuracy goes up for each epoch
#       - NOTE: "backbone.pth" file is generated within that net folder if n_epochs is given! -> so when you run it next time, it uses that .pth file instead to load all the weights
#       - But with .pth, the size of onnx increases

# Also, the more you train, the more stuffs get added to .stats folder -> resulting in increase in size
