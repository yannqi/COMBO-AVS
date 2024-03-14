dataset_root=${2:-'AVS_dataset/AVSBench_object/Single-source/'}
export DETECTRON2_DATASETS=$dataset_root
export CUDA_VISIBLE_DEVICES=0
python pred.py \
    --num-gpus 1 \
    --config-file configs/avs_s4/Test_COMBO_R50_bs8_90k.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only \

