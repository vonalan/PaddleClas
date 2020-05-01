export PYTHONPATH=/data2/lijinde/Programs/PaddleClas:$PYTHONPATH

prefix=ResNet50_vd_10w_pretrained

python tools/infer/predict.py \
    -i=${1} \
    -m=deploy/${prefix}/model \
    -p=deploy/${prefix}/params \
    --use_gpu=0 \
    --use_tensorrt=False
