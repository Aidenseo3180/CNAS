python ee_train.py --dataset imagenette --model eemobilenetv3 --device 0 --resolution 160 --model_path results/search_edanas_imagenette/iter_25/net_6/net_6.subnet --output_path results/prova_search_edanas_imagenette/iter_25/net_6 --pretrained --supernet_path NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 --batch_size 128 --mmax 100 --top1min 0.1 --method joint --val_split 0.1 --w_gamma 1.0 --w_beta 1.0 --w_alpha 1.0 --backbone_epochs 5 --warmup_ee_epochs 5 --ee_epochs 5 --n_workers 4