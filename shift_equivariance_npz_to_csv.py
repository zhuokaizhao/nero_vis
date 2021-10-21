# convert part of saved npz results to csv
import os
import numpy as np

# general information
image_size = (128, 128)
desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
# model type and jittering levels for the custom-trained models
model_types = ['normal', 'si', 'pt']
# model_types = ['normal']
all_jitters = [0, 20, 40, 60, 80, 100]

convert_test = True
convert_single_test = True

if convert_test:
    ################### For test #################
    # convert the general test result from npz to csv
    output_dir = 'shift_equivariance_d3_vis/data/test'
    positive_trans = [1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]
    negative_trans = list(reversed(positive_trans))
    negative_trans = [-x for x in negative_trans]
    x_translation = negative_trans + [0] + positive_trans
    y_translation = negative_trans + [0] + positive_trans

    # convert the test results
    for model_type in model_types:

        if model_type == 'pt':
            cur_npz_path = f'output/test/{model_type}'
            cur_npz_name = f'faster_rcnn_resnet50_test_{model_type}.npz'
            cur_npz_path = os.path.join(cur_npz_path, cur_npz_name)

            # load the npz file
            test_results = np.load(cur_npz_path)
            for name in test_results.keys():
                cur_result = test_results[name]

                # convert to 7-column format (x_tran y_tran) + scalar result for each class
                cur_csv_content = []
                for y, y_tran in enumerate(y_translation):
                    for x, x_tran in enumerate(x_translation):
                        cur_csv_content.append((x_tran,
                                                y_tran,
                                                cur_result[y, x, 0],
                                                cur_result[y, x, 1],
                                                cur_result[y, x, 2],
                                                cur_result[y, x, 3],
                                                cur_result[y, x, 4]))

                cur_csv_path = os.path.join(output_dir, f"{model_type}_{name.split('_')[-1]}.csv")
                np.savetxt(cur_csv_path, cur_csv_content, delimiter=',', fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'], header='x_tran,y_tran,car,bottle,cup,chair,book', comments='')
                print(f'\n{model_type} {name} csv has been saved to {cur_csv_path}\n')

        else:
            for jitter in all_jitters:
                cur_npz_path = f'output/test/{model_type}'
                cur_npz_name = f'faster_rcnn_resnet50_test_{jitter}-jittered_{model_type}.npz'
                cur_npz_path = os.path.join(cur_npz_path, cur_npz_name)

                # load the npz file
                test_results = np.load(cur_npz_path)
                for name in test_results.keys():
                    cur_result = test_results[name]

                    # convert to 7-column format (x_tran y_tran) + scalar result for each class
                    cur_csv_content = []
                    for y, y_tran in enumerate(y_translation):
                        for x, x_tran in enumerate(x_translation):
                            cur_csv_content.append((x_tran,
                                                    y_tran,
                                                    cur_result[y, x, 0],
                                                    cur_result[y, x, 1],
                                                    cur_result[y, x, 2],
                                                    cur_result[y, x, 3],
                                                    cur_result[y, x, 4]))

                    cur_csv_path = os.path.join(output_dir, f"{model_type}_{jitter}_{name.split('_')[-1]}.csv")
                    np.savetxt(cur_csv_path, cur_csv_content, delimiter=',', fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f'], header='x_tran,y_tran,car,bottle,cup,chair,book', comments='')
                    print(f'\n{model_type} {jitter} {name} csv has been saved to {cur_csv_path}\n')



################### For single test #################
if convert_single_test:

    model_types = ['normal', 'si']

    # convert the single test result from npz to csv
    x_translation = list(range(-image_size[0]//2, image_size[0]//2))
    y_translation = list(range(-image_size[1]//2, image_size[1]//2))
    # top-n results that we are keeping for the single-test cases
    num_keep = 3

    # controlling for the single-test cases
    convert_target_id = True
    convert_target_bb = True
    convert_pre_trained = True
    convert_custom_trained = True
    convert_pred_id = True
    convert_pred_bb = True
    convert_confidence = True
    convert_iou = True
    convert_precision = True

    for i, cur_class in enumerate(desired_classes):

        # this path is only for getting the ground truths (different model types share the same ground truths)
        npz_path = f'output/single_test/normal/{cur_class}/faster_rcnn_resnet50_single-test_0-jittered_normal_0.5.npz'

        if convert_target_id:
            # generate pred and target labels
            target_id_csv_path = f'shift_equivariance_d3_vis/data/single_test/target_id/{cur_class}/target_id.csv'

            # load the npz file
            target_id = np.load(npz_path)['all_target_label']
            # convert to 3-column format (x_tran y_tran id)
            pred_id_csv_content = []
            for y, y_tran in enumerate(y_translation):
                for x, x_tran in enumerate(x_translation):
                    pred_id_csv_content.append((x_tran, y_tran, target_id[y, x]))

            np.savetxt(target_id_csv_path, pred_id_csv_content, delimiter=',', fmt=['%d', '%d', '%d'], header='x_tran,y_tran,id', comments='')
            print(f'\ntarget id csv has been saved to {target_id_csv_path}\n')

        if convert_target_bb:
            # generate pred and target labels
            target_bb_csv_path = f'shift_equivariance_d3_vis/data/single_test/target_bb/{cur_class}/target_bb.csv'

            # load the npz file
            target_bb = np.load(npz_path)['all_target_bb']
            # convert to 7-column format (x_tran y_tran x y width height)
            target_bb_csv_content = []
            for y, y_tran in enumerate(y_translation):
                for x, x_tran in enumerate(x_translation):
                    x_min = target_bb[y, x, 0]
                    y_min = target_bb[y, x, 1]
                    width = target_bb[y, x, 2] - target_bb[y, x, 0]
                    height = target_bb[y, x, 3] - target_bb[y, x, 1]
                    target_bb_csv_content.append((x_tran, y_tran, x_min, y_min, width, height))

            np.savetxt(target_bb_csv_path, target_bb_csv_content, delimiter=',', fmt=['%d', '%d', '%f', '%f', '%f', '%f'], header='x_tran,y_tran,x_min,y_min,width,height', comments='')
            print(f'\ntarget bounding box csv has been saved to {target_bb_csv_path}\n')

        # pre-trained model
        if convert_pre_trained:
            # npz path of the pretrained model results
            npz_path = f'output/single_test/pt/{cur_class}/faster_rcnn_resnet50_single-test_pt_0.5.npz'

            if convert_pred_id:
                for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                    # generate pred and target labels
                    pred_id_csv_path = f'shift_equivariance_d3_vis/data/single_test/pretrained_pred_id/{cur_class}/pred_id_{mode}_pretrained.csv'

                    # load the npz file
                    pred_id = np.load(npz_path)[f'all_pred_label_{mode}']

                    # convert to 3-column format (x_tran y_tran id)
                    pred_id_csv_content = []
                    for y, y_tran in enumerate(y_translation):
                        for x, x_tran in enumerate(x_translation):
                            cur_content = [x_tran, y_tran]
                            for i in range(num_keep):
                                cur_content.append(pred_id[y, x, i])

                            pred_id_csv_content.append(cur_content)

                    np.savetxt(pred_id_csv_path, pred_id_csv_content, delimiter=',', fmt=['%d', '%d', '%d', '%d', '%d'], header='x_tran,y_tran,id,id1,id2', comments='')
                    print(f'\nPretrained pred id csv has been saved to {pred_id_csv_path}\n')

            if convert_pred_bb:
                for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                    # generate pred and target labels
                    pred_bb_csv_path = f'shift_equivariance_d3_vis/data/single_test/pretrained_pred_bb/{cur_class}/pred_bb_{mode}_pretrained.csv'

                    # load the npz file
                    pred_bb = np.load(npz_path)[f'all_pred_bb_{mode}']
                    # each cell in pred bb has x_min, y_min, x_max, y_max
                    # convert to 7-column format (x_tran y_tran x y width height)
                    pred_bb_csv_content = []
                    for y, y_tran in enumerate(y_translation):
                        for x, x_tran in enumerate(x_translation):
                            cur_content = [x_tran, y_tran]
                            for i in range(num_keep):
                                x_min = pred_bb[y, x, i*4+0]
                                y_min = pred_bb[y, x, i*4+1]
                                width = pred_bb[y, x, i*4+2] - pred_bb[y, x, i*4+0]
                                height = pred_bb[y, x, i*4+3] - pred_bb[y, x, i*4+1]
                                cur_content.append(x_min)
                                cur_content.append(y_min)
                                cur_content.append(width)
                                cur_content.append(height)

                            pred_bb_csv_content.append(cur_content)

                    np.savetxt(pred_bb_csv_path, pred_bb_csv_content, delimiter=',',
                                fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'],
                                header='x_tran,y_tran,x_min,y_min,width,height,x_min1,y_min1,width1,height1,x_min2,y_min2,width2,height2',
                                comments='')
                    print(f'\nPretrained pred bounding box csv has been saved to {pred_bb_csv_path}\n')

            if convert_iou:
                for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                    # generate cq csv's
                    iou_csv_path = f'shift_equivariance_d3_vis/data/single_test/pretrained_iou/{cur_class}/iou_{mode}_pretrained.csv'
                    # load the npz file
                    iou = np.load(npz_path)[f'all_iou_{mode}']
                    # convert to 3-column format (x_tran y_tran quantity)
                    iou_csv_content = []
                    for y, y_tran in enumerate(y_translation):
                        for x, x_tran in enumerate(x_translation):
                            cur_content = [x_tran, y_tran]
                            for i in range(num_keep):
                                cur_content.append(iou[y, x, i])

                            iou_csv_content.append(cur_content)

                    np.savetxt(iou_csv_path, iou_csv_content, delimiter=',',
                                fmt=['%d', '%d', '%f', '%f', '%f'],
                                header='x_tran,y_tran,iou,iou1,iou2',
                                comments='')
                    print(f'\nPretrained iou csv has been saved to {iou_csv_path}\n')

            if convert_confidence:
                for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                    # generate cq csv's
                    confidence_csv_path = f'shift_equivariance_d3_vis/data/single_test/pretrained_confidence/{cur_class}/confidence_{mode}_pretrained.csv'
                    # load the npz file
                    confidence = np.load(npz_path)[f'all_confidence_{mode}']

                    # convert to 3-column format (x_tran y_tran quantity)
                    confidence_csv_content = []
                    for y, y_tran in enumerate(y_translation):
                        for x, x_tran in enumerate(x_translation):
                            cur_content = [x_tran, y_tran]
                            for i in range(num_keep):
                                cur_content.append(confidence[y, x, i])

                            confidence_csv_content.append(cur_content)

                    np.savetxt(confidence_csv_path, confidence_csv_content, delimiter=',',
                                fmt=['%d', '%d', '%f', '%f', '%f'],
                                header='x_tran,y_tran,confidence,confidence1,confidence2',
                                comments='')
                    print(f'\nPretrained confidence csv has been saved to {confidence_csv_path}\n')

            if convert_precision:
                # generate cq csv's
                precision_csv_path = f'shift_equivariance_d3_vis/data/single_test/pretrained_precision/{cur_class}/precision_pretrained.csv'
                # load the npz file
                precision = np.load(npz_path)['all_precision']
                # convert to 3-column format (x_tran y_tran quantity)
                precision_csv_content = []
                for y, y_tran in enumerate(y_translation):
                    for x, x_tran in enumerate(x_translation):
                        precision_csv_content.append((x_tran, y_tran, precision[y, x]))

                np.savetxt(precision_csv_path, precision_csv_content, delimiter=',', fmt=['%d', '%d', '%f'], header='x_tran,y_tran,precision', comments='')
                print(f'\nPretrained precision csv has been saved to {precision_csv_path}\n')

        # custom-trained models
        if convert_custom_trained:
            for model_type in model_types:
                for jitter in all_jitters:
                    npz_path = f'output/single_test/{model_type}/{cur_class}/faster_rcnn_resnet50_single-test_{jitter}-jittered_{model_type}_0.5.npz'

                    if convert_pred_id:
                        for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                            # generate pred and target labels
                            pred_id_csv_path = f'shift_equivariance_d3_vis/data/single_test/{model_type}_pred_id/{cur_class}/pred_id_{mode}_{model_type}_{jitter}.csv'

                            # load the npz file
                            pred_id = np.load(npz_path)[f'all_pred_label_{mode}']
                            # convert to 3-column format (x_tran y_tran id)
                            pred_id_csv_content = []
                            for y, y_tran in enumerate(y_translation):
                                for x, x_tran in enumerate(x_translation):
                                    cur_content = [x_tran, y_tran]
                                    for i in range(num_keep):
                                        cur_content.append(pred_id[y, x, i])

                                    pred_id_csv_content.append(cur_content)

                            np.savetxt(pred_id_csv_path, pred_id_csv_content, delimiter=',', fmt=['%d', '%d', '%d', '%d', '%d'], header='x_tran,y_tran,id,id1,id2', comments='')
                            print(f'\n{model_type} pred id csv has been saved to {pred_id_csv_path}\n')

                    if convert_pred_bb:
                        for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                            # generate pred and target labels
                            pred_bb_csv_path = f'shift_equivariance_d3_vis/data/single_test/{model_type}_pred_bb/{cur_class}/pred_bb_{mode}_{model_type}_{jitter}.csv'

                            # load the npz file
                            pred_bb = np.load(npz_path)[f'all_pred_bb_{mode}']
                            # each cell in pred bb has x_min, y_min, x_max, y_max
                            # convert to 7-column format (x_tran y_tran x y width height)
                            pred_bb_csv_content = []
                            for y, y_tran in enumerate(y_translation):
                                for x, x_tran in enumerate(x_translation):
                                    cur_content = [x_tran, y_tran]
                                    for i in range(num_keep):
                                        x_min = pred_bb[y, x, i*4+0]
                                        y_min = pred_bb[y, x, i*4+1]
                                        width = pred_bb[y, x, i*4+2] - pred_bb[y, x, i*4+0]
                                        height = pred_bb[y, x, i*4+3] - pred_bb[y, x, i*4+1]
                                        cur_content.append(x_min)
                                        cur_content.append(y_min)
                                        cur_content.append(width)
                                        cur_content.append(height)

                                    pred_bb_csv_content.append(cur_content)

                            np.savetxt(pred_bb_csv_path, pred_bb_csv_content, delimiter=',',
                                        fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f'],
                                        header='x_tran,y_tran,x_min,y_min,width,height,x_min1,y_min1,width1,height1,x_min2,y_min2,width2,height2',
                                        comments='')
                            print(f'\n{model_type} pred bounding box csv has been saved to {pred_bb_csv_path}\n')

                    if convert_iou:
                        for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                            # generate cq csv's
                            iou_csv_path = f'shift_equivariance_d3_vis/data/single_test/{model_type}_iou/{cur_class}/iou_{mode}_{model_type}_{jitter}.csv'
                            # load the npz file
                            iou = np.load(npz_path)[f'all_iou_{mode}']
                            # convert to 3-column format (x_tran y_tran quantity)
                            iou_csv_content = []
                            for y, y_tran in enumerate(y_translation):
                                for x, x_tran in enumerate(x_translation):
                                    cur_content = [x_tran, y_tran]
                                    for i in range(num_keep):
                                        cur_content.append(iou[y, x, i])

                                    iou_csv_content.append(cur_content)

                            np.savetxt(iou_csv_path, iou_csv_content, delimiter=',',
                                        fmt=['%d', '%d', '%f', '%f', '%f'],
                                        header='x_tran,y_tran,iou,iou1,iou2',
                                        comments='')
                            print(f'\n{model_type} iou csv has been saved to {iou_csv_path}\n')

                    if convert_confidence:
                        for mode in ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr']:
                            # generate cq csv's
                            confidence_csv_path = f'shift_equivariance_d3_vis/data/single_test/{model_type}_confidence/{cur_class}/confidence_{mode}_{model_type}_{jitter}.csv'
                            # load the npz file
                            confidence = np.load(npz_path)[f'all_confidence_{mode}']
                            # convert to 3-column format (x_tran y_tran quantity)
                            confidence_csv_content = []
                            for y, y_tran in enumerate(y_translation):
                                for x, x_tran in enumerate(x_translation):
                                    cur_content = [x_tran, y_tran]
                                    for i in range(num_keep):
                                        cur_content.append(confidence[y, x, i])

                                    confidence_csv_content.append(cur_content)

                            np.savetxt(confidence_csv_path, confidence_csv_content, delimiter=',',
                                        fmt=['%d', '%d', '%f', '%f', '%f'],
                                        header='x_tran,y_tran,confidence,confidence1,confidence2',
                                        comments='')
                            print(f'\n{model_type} confidence csv has been saved to {confidence_csv_path}\n')

                    if convert_precision:
                        # generate cq csv's
                        precision_csv_path = f'shift_equivariance_d3_vis/data/single_test/{model_type}_precision/{cur_class}/precision_{model_type}_{jitter}.csv'
                        # load the npz file
                        precision = np.load(npz_path)['all_precision']
                        # convert to 3-column format (x_tran y_tran quantity)
                        precision_csv_content = []
                        for y, y_tran in enumerate(y_translation):
                            for x, x_tran in enumerate(x_translation):
                                precision_csv_content.append((x_tran, y_tran, precision[y, x]))

                        np.savetxt(precision_csv_path, precision_csv_content, delimiter=',', fmt=['%d', '%d', '%f'], header='x_tran,y_tran,precision', comments='')
                        print(f'\n{model_type} precision csv has been saved to {precision_csv_path}\n')
