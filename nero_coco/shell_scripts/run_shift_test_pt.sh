#!/bin/bash
python test.py --purpose shift-equivariance --mode test --data_name coco --batch_size 32 --data_config config/shift_equivariance/COCO/test/custom_test_all_shifts.data --output_dir output/test/pt/ --model_type pt -v
