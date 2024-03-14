dataset_root=${2:-'AVS_dataset/AVSBench_object/Multi-sources/'}
export DETECTRON2_DATASETS=$dataset_root
export CUDA_VISIBLE_DEVICES=1
python pred.py \
    --num-gpus 1 \
    --config-file configs/avs_ms3/Test_COMBO_PVTV2B5_bs8_20k.yaml \
    --dist-url tcp://0.0.0.0:47772 \
    --eval-only \
    