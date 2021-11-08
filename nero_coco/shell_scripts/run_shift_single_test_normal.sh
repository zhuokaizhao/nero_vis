#!/bin/bash
# for jittering in '0' '20' '40' '60' '80' '100'
# for jittering in '0' '20' '40'
for jittering in '33'
do
    echo "jittering: ${jittering}"
    for class_name in 'car' 'bottle' 'cup' 'chair' 'book'
    # for class_name in 'book'
    # for class_name in 'car'
    do
        echo "class name: ${class_name}"
        if [ "$jittering" = "0" ];
        then
            python test.py --purpose shift-equivariance --mode single-test --data_name coco --batch_size 1 --data_config config/shift_equivariance/COCO/single_test/custom_single_test_${class_name}.data --percentage ${jittering} --model_path model/object_${jittering}-jittered/faster_rcnn_resnet50_${jittering}-jittered_ckpt_50.pth --output_dir output/single_test/normal/${class_name}/ --model_type normal -v
        else
            python test.py --purpose shift-equivariance --mode single-test --data_name coco --batch_size 1 --data_config config/shift_equivariance/COCO/single_test/custom_single_test_${class_name}.data --percentage ${jittering} --model_path model/object_${jittering}-jittered/faster_rcnn_resnet50_${jittering}-jittered_ckpt_100.pth --output_dir output/single_test/normal/${class_name}/ --model_type normal -v
        fi
    done
done
