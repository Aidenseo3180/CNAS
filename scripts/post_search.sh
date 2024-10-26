# Look for the n architectures in the archive at iteration iter with the best trade-off between the first and second objectives

# dataset=cifar10 (change num classes accordingly)

# NOTE: sec_obj throws an error, so exclude it for now. Later, this first_obj will be the model size

first_obj=top1
# sec_obj=c_params
iter=20

python post_search.py \
  --supernet_path ./NasSearchSpace/ofa/supernets/ofa_mbv3_d234_e346_k357_w1.0 \
  --get_archive --n 3 --n_classes 10 \
  --save results/search_path/final \
  --expr results/search_path/iter_$iter.stats \
  --first_obj $first_obj # --sec_obj $sec_obj 

# --n means we're going to find the top 3 subnets with best first_obj