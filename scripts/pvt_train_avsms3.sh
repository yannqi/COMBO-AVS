dataset_root=${2:-'AVS_dataset/AVSBench_object/Multi-sources/'}
export DETECTRON2_DATASETS=$dataset_root
export CUDA_VISIBLE_DEVICES=0
python train_net.py \
    --num-gpus 1 \
    --config-file configs/avs_ms3/COMBO_PVTV2B5_bs8_20k.yaml \
    --dist-url tcp://0.0.0.0:47773 \
