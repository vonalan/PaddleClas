export PYTHONPATH=/data2/{1}/Programs/PaddleClas:$PYTHONPATH

prefix=ResNet50_vd_10w_pretrained

python tools/infer/predict.py \
    -i=${2} \
    -m=deploy/${prefix}/model \
    -p=deploy/${prefix}/params \
    --use_gpu=1 \
    --use_tensorrt=False \
