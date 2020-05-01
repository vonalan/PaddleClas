export PYTHONPATH=/data2/lijinde/Programs/PaddleClas:$PYTHONPATH

prefix=ResNet50_vd_10w_pretrained

python tools/export_model.py \
    --m=ResNet50_vd \
    --p=pretrained/${prefix} \
    --o=deploy/${prefix} \
    --include_top=0