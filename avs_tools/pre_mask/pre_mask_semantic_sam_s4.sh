export CUDA_VISIBLE_DEVICES=0
python tools/pre_mask/make_SAM_mask.py \
    --sam_type 'semantic_sam' \
    --data_name 's4' \
    --output 'AVS_dataset/pre_SemanticSAM_mask' \
    --split $1 \
# split : train, val, test 