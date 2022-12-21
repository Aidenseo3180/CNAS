cd ../nsganetv2

python train_cifar.py --data ../data/cifar10 --dataset cifar10 \
    --epochs 20 \
    --cutout --autoaugment \
    --evaluate \
    --model tinynsganet \
    --model-config ../results/20221220-155540cifar10adapt_tinynsganet40/net_flops@7.config \
    --initial-checkpoint ../results/20221220-155540cifar10adapt_tinynsganet40/net_flops@7.best \
    --drop 0.2 --drop-path 0.2 \
    --img-size 40