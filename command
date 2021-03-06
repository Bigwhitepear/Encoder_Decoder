CUDA_VISIBLE_DEVICES=6 nohup python -u run.py -batch 256 -gat_layer 2 -gat_drop 0.3 \
-gat_alpha 0.2 > res/31_wn18rr.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u run.py -batch 256 -gat_layer 2 -gat_drop 0.1 -hid_drop 0.1 \
-gat_alpha 0.1 -hid_drop2 0.1 -data WN18RR -way s > res/34_wn18rr_layer2.log &

CUDA_VISIBLE_DEVICES=3 nohup python -u run.py -batch 256 -gat_layer 3 -gat_drop 0.1 -hid_drop 0.1 \
-gat_alpha 0.1 -hid_drop2 0.1 -data WN18RR -way s > res/34_wn18rr_layer3.log &

CUDA_VISIBLE_DEVICES=4 nohup python -u run.py -batch 256 -gat_layer 4 -gat_drop 0.1 -hid_drop 0.1 \
-gat_alpha 0.1 -hid_drop2 0.1 -data WN18RR -way s > res/34_wn18rr_layer4.log &

CUDA_VISIBLE_DEVICES=0 nohup python -u run.py -batch 128 -gat_layer 2 -gat_drop 0.1 -hid_drop 0.1 \
-gat_alpha 0.1 -hid_drop2 0.1 -data FB15k-237 -way s > res/34_fb15k237_layer2.log &

CUDA_VISIBLE_DEVICES=5 nohup python -u run.py -batch 256 -gat_layer 1 -gat_drop 0.1 -hid_drop 0.1 \
-gat_alpha 0.1 -hid_drop2 0.1 -data WN18RR -way s > res/34_wn18rr_layer1.log &