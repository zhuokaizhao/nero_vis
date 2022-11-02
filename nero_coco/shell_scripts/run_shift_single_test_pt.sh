#!/bin/bash
for class_name in 'car' 'bottle' 'cup' 'chair' 'book'
# for class_name in 'cup'
do
    echo "class name: ${class_name}"
    python test.py --purpose shift-equivariance --mode single-test --data_name coco --batch_size 1 --data_config config/shift_equivariance/COCO/single_test/custom_single_test_${class_name}.data --output_dir output/single_test/pt/${class_name}/ --model_type pt -v
done
