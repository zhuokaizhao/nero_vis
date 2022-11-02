#!/bin/bash
# for jittering in '0' '20' '40' '60' '80' '100'
# for jittering in '0'
# for jittering in '20'
# for jittering in '40'
# for jittering in '60'
# for jittering in '80'
for jittering in '66'
# for jittering in '0' '20' '40'
do
    echo "jittering: ${jittering}"
    if [ "$jittering" = "0" ];
    then
        python test.py --purpose shift-equivariance --mode test --data_name coco --batch_size 32 --data_config config/shift_equivariance/COCO/test/custom_test_all_shifts.data --percentage ${jittering} --model_path model/object_${jittering}-jittered/faster_rcnn_resnet50_${jittering}-jittered_ckpt_50.pth --output_dir output/test/ --model_type normal -v
    else
        python test.py --purpose shift-equivariance --mode test --data_name coco --batch_size 32 --data_config config/shift_equivariance/COCO/test/custom_test_all_shifts.data --percentage ${jittering} --model_path model/object_${jittering}-jittered/faster_rcnn_resnet50_${jittering}-jittered_ckpt_105.pth --output_dir output/test/ --model_type normal -v
    fi
done
