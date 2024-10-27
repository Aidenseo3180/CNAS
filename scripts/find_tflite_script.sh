# Basically the same as the script that you ran for finding subnets
subnet_path=results/search_path/iter_0/net_1/net_1.subnet
output_path=results/search_path/iter_0/net_1

# Select the filename you would like for onnx
onnx_filename=cnas_multiexit

python find_tflite.py --n_workers 4 \
              --dataset cifar10 \
              --model_path $subnet_path \
              --output_path $output_path \
              --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --pretrained  \
              --method bernulli --support_set --tune_epsilon\
              --val_split 0.1 \
              --device 0 \
              --backbone_epochs 0 \
              --warmup_ee_epochs 0 --ee_epochs 0 \
              --w_alpha 1.0 --w_beta 1.0 --w_gamma 1.0 \
              --mmax 2.7 --top1min 0.65 \
              --onnx_filename $onnx_filename \

# NOTE: --device 0 means cuda:0. If you're gpu has different name, modify it manually