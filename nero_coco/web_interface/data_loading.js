
// load all csv for the single test
function load_test_data(plot) {

    // input arguments
    var model_types = ['normal', 'si', 'pt'];
    var result_categories = ['precision', 'recall', 'AP', 'f1'];

    // normal and si result paths
    var all_result_paths = [];

    // construct the paths
    for (let t = 0; t < model_types.length; t++) {
        if (t == 2) {
            for (let j = 0; j < result_categories.length; j++) {
                // current path
                cur_path = data_dir.concat('test/', model_types[t], '_', result_categories[j], '.csv');
                all_result_paths.push(cur_path);
            }
        }
        else {
            for (let i = 0; i < result_jittering_levels.length; i++) {
                for (let j = 0; j < result_categories.length; j++) {
                    // current path
                    cur_path = data_dir.concat('test/', model_types[t], '_', result_jittering_levels[i], '_', result_categories[j], '.csv');
                    all_result_paths.push(cur_path);
                }
            }
        }
    }

    // load result data
    var all_result_data = [];
    var result_queue = d3.queue();
    // queueing result-loading tasks
    all_result_paths.forEach(function(filename) {
        result_queue.defer(d3.csv, filename);
    });

    // temporary data container
    var all_jitter_precision = {};
    var all_jitter_recall = {};
    var all_jitter_AP = {};
    var all_jitter_f1 = {};

    // processing the result queues
    result_queue.awaitAll(function(error, all_result_data) {
        if (error) {
            throw error;
        }
        else {
            // each model type has 4*len(result_jittering_levels) files
            var num_datasets = 4*result_jittering_levels.length;

            for (let t = 0; t < model_types.length; t++) {

                // clear the container
                all_jitter_precision = {};
                all_jitter_recall = {};
                all_jitter_AP = {};
                all_jitter_f1 = {};

                if (t == 2) {
                    for (let j = 0; j < result_categories.length; j++) {

                        // load the data into container
                        if (j == 0) {
                            all_jitter_precision['all'] = all_result_data[2*num_datasets+j];
                        }
                        else if (j == 1) {
                            all_jitter_recall['all'] = all_result_data[2*num_datasets+j];
                        }
                        else if (j == 2) {
                            all_jitter_AP['all'] = all_result_data[2*num_datasets+j];
                        }
                        else if (j == 3) {
                            all_jitter_f1['all'] = all_result_data[2*num_datasets+j];
                        }
                    }
                }
                else {
                    for (let i = 0; i < result_jittering_levels.length; i++) {
                        for (let j = 0; j < result_categories.length; j++) {
                            var cur_jitter_level = result_jittering_levels[i];
                            // load the data into container
                            if (j == 0) {
                                all_jitter_precision[cur_jitter_level] = all_result_data[t*num_datasets+i*4+j];
                            }
                            else if (j == 1) {
                                all_jitter_recall[cur_jitter_level] = all_result_data[t*num_datasets+i*4+j];
                            }
                            else if (j == 2) {
                                all_jitter_AP[cur_jitter_level] = all_result_data[t*num_datasets+i*4+j];
                            }
                            else if (j == 3) {
                                all_jitter_f1[cur_jitter_level] = all_result_data[t*num_datasets+i*4+j];
                            }
                        }
                    }
                }

                // assign results
                if (t == 0) {
                    normal_test_data['precision'] = all_jitter_precision;
                    normal_test_data['recall'] = all_jitter_recall;
                    normal_test_data['AP'] = all_jitter_AP;
                    normal_test_data['f1'] = all_jitter_f1;
                }
                else if (t == 1) {
                    si_test_data['precision'] = all_jitter_precision;
                    si_test_data['recall'] = all_jitter_recall;
                    si_test_data['AP'] = all_jitter_AP;
                    si_test_data['f1'] = all_jitter_f1;
                }
                else if (t == 2) {
                    pt_test_data['precision'] = all_jitter_precision;
                    pt_test_data['recall'] = all_jitter_recall;
                    pt_test_data['AP'] = all_jitter_AP;
                    pt_test_data['f1'] = all_jitter_f1;
                }
            }

            // plot the graph for the first time
            if (plot) {
                plot_test_heatmap('normal', normal_test_data, normal_svg);
                plot_test_heatmap('si', si_test_data, si_svg);
                plot_test_heatmap('pt', pt_test_data, pt_svg);
            }

            console.log('Test results data loaded successfully.');
        }
    });
}

// load all csv for the single test
function load_single_test_data() {

    // input arguments
    var target_categories = ['id', 'bb'];
    var result_categories = ['pred_id', 'pred_bb', 'confidence', 'iou', 'precision'];
    var result_sort_methods = ['sorted_by_conf', 'sorted_by_iou', 'sorted_by_conf_iou', 'sorted_by_conf_iou_pr'];

    // cutout image position paths
    var all_cutout_pos_paths = [];
    // target (ground truths) paths
    var all_target_paths = [];
    // normal and si model results paths
    var all_normal_result_paths = [];
    var all_si_result_paths = [];
    var all_pretrained_result_paths = [];

    // construct all the data paths for all object classes
    for (let i = 0; i < desired_classes.length; i++) {
        var cur_class = desired_classes[i];
        // cutout position paths
        // example: './images/image_cutout_pos_899_0.csv'
        cur_pos_path = data_dir.concat('images/image_cutout_pos_', cur_class, '.csv');
        all_cutout_pos_paths.push(cur_pos_path);

        // all target paths
        for (let a = 0; a < target_categories.length; a++) {
            var cur_category = target_categories[a];
            // construct the path
            // example: './target_id/image_899/target_id.csv'
            cur_path = data_dir.concat('single_test/target_', cur_category, '/', cur_class, '/target_', cur_category, '.csv');
            all_target_paths.push(cur_path);
        }

        // all models results paths (normal, si and pretrained)
        for (let a = 0; a < result_categories.length; a++) {
            var cur_category = result_categories[a];
            // precision has no sorting method
            if (cur_category == 'precision') {
                for (let c = 0; c < result_jittering_levels.length; c++) {
                    var cur_jitter_level = result_jittering_levels[c];
                    // construct the path
                    // example: './normal_precision/car/precision_si_100.csv'
                    cur_path = data_dir.concat('single_test/normal_', cur_category, '/', cur_class,
                                                '/', cur_category, '_normal_', cur_jitter_level, '.csv');
                    all_normal_result_paths.push(cur_path);
                    // cur_path = data_dir.concat('single_test/si_', cur_category, '/', cur_class,
                    //                             '/', cur_category, '_si_', cur_jitter_level, '.csv');
                    // all_si_result_paths.push(cur_path);
                }
                cur_path = data_dir.concat('single_test/pretrained_', cur_category, '/', cur_class,
                                            '/', cur_category, '_pretrained.csv');
                all_pretrained_result_paths.push(cur_path);
            }
            else {
                for (let b = 0; b < result_sort_methods.length; b++) {
                    var cur_sort = result_sort_methods[b];
                    // normal and si, which have different jittering levels
                    for (let c = 0; c < result_jittering_levels.length; c++) {
                        cur_jitter_level = result_jittering_levels[c];
                        // construct the path
                        // example: './normal_pred_id/image_899/pred_id_sorted_by_conf_normal_0.csv'
                        var cur_path = data_dir.concat('single_test/normal_', cur_category, '/', cur_class,
                                                        '/', cur_category, '_', cur_sort, '_normal_', cur_jitter_level, '.csv');
                        all_normal_result_paths.push(cur_path);
                        cur_path = data_dir.concat('single_test/si_', cur_category, '/', cur_class,
                                                    '/', cur_category, '_', cur_sort, '_si_', cur_jitter_level, '.csv');
                        all_si_result_paths.push(cur_path);
                    }

                    // pretrained, which does not have different jittering levels
                    var cur_path = data_dir.concat('single_test/pretrained_', cur_category, '/', cur_class,
                                                    '/', cur_category, '_', cur_sort, '_pretrained.csv');
                    all_pretrained_result_paths.push(cur_path);
                }
            }
        }
    }

    // combine both models' output paths
    var all_result_paths = all_normal_result_paths.concat(all_si_result_paths);
    all_result_paths = all_result_paths.concat(all_pretrained_result_paths);

    // load cutout position data
    var loaded_cutout_pos_data = [];
    cutout_pos_queue = d3.queue();
    // queueing cutout-pos-loading tasks
    all_cutout_pos_paths.forEach(function(filename) {
        cutout_pos_queue.defer(d3.csv, filename);
    });
    // processing the cutout pos queue
    cutout_pos_queue.awaitAll(function(error, loaded_cutout_pos_data) {
        if (error) {
            throw error;
        }
        else {
            cutout_pos_data = loaded_cutout_pos_data;
            console.log('Cutout position data loaded successfully.');
        }
    });

    // load target (ground truth) data
    all_target_data = [];
    target_queue = d3.queue();
    // queueing csv-loading tasks
    all_target_paths.forEach(function(filename) {
        target_queue.defer(d3.csv, filename);
    });
    // processing the target queue
    target_queue.awaitAll(function(error, all_target_data) {
        if (error) {
            throw error;
        }
        else {
            // divide the id and bb
            for (let i = 0; i < 2*desired_classes.length; i++) {
                if (i % 2 == 0) {
                    target_id_data.push(all_target_data[i]);
                }
                else {
                    target_bb_data.push(all_target_data[i]);
                }
            }
            console.log('Target id and bb data loaded successfully.')
        }
    });

    // load result data
    var all_result_data = [];
    var result_queue = d3.queue();
    // queueing result-loading tasks
    all_result_paths.forEach(function(filename) {
        result_queue.defer(d3.csv, filename);
    });
    // temporary data containers
    // sorted based on conf
    var cur_pred_id_data_sorted_by_conf = {};
    var cur_pred_bb_data_sorted_by_conf = {};
    var cur_confidence_data_sorted_by_conf = {};
    var cur_iou_data_sorted_by_conf = {};
    // sorted based on iou
    var cur_pred_id_data_sorted_by_iou = {};
    var cur_pred_bb_data_sorted_by_iou = {};
    var cur_confidence_data_sorted_by_iou = {};
    var cur_iou_data_sorted_by_iou = {};
    // sorted based on conf*iou
    var cur_pred_id_data_sorted_by_conf_iou = {};
    var cur_pred_bb_data_sorted_by_conf_iou = {};
    var cur_confidence_data_sorted_by_conf_iou = {};
    var cur_iou_data_sorted_by_conf_iou = {};
    // sorted based on conf*iou*pr
    var cur_pred_id_data_sorted_by_conf_iou_pr = {};
    var cur_pred_bb_data_sorted_by_conf_iou_pr = {};
    var cur_confidence_data_sorted_by_conf_iou_pr = {};
    var cur_iou_data_sorted_by_conf_iou_pr = {};
    var cur_precision_data = {};
    // processing the result queues
    result_queue.awaitAll(function(error, all_result_data) {
        if (error) {
            throw error;
        }
        else {
            // k=0 means normal, k=1 means shift-invariant, k=2 means pretrained model results
            // var all_methods = [0, 2]
            for (let k = 0; k < 3; k++) {
            // for (const k of all_methods) {
                // for each image
                for (let i = 0; i < desired_classes.length; i++) {
                    // empty containers
                    cur_pred_id_data_sorted_by_conf = {};
                    cur_pred_bb_data_sorted_by_conf = {};
                    cur_confidence_data_sorted_by_conf = {};
                    cur_iou_data_sorted_by_conf = {};
                    cur_pred_id_data_sorted_by_iou = {};
                    cur_pred_bb_data_sorted_by_iou = {};
                    cur_confidence_data_sorted_by_iou = {};
                    cur_iou_data_sorted_by_iou = {};
                    cur_pred_id_data_sorted_by_conf_iou = {};
                    cur_pred_bb_data_sorted_by_conf_iou = {};
                    cur_confidence_data_sorted_by_conf_iou = {};
                    cur_iou_data_sorted_by_conf_iou = {};
                    cur_pred_id_data_sorted_by_conf_iou_pr = {};
                    cur_pred_bb_data_sorted_by_conf_iou_pr = {};
                    cur_confidence_data_sorted_by_conf_iou_pr = {};
                    cur_iou_data_sorted_by_conf_iou_pr = {};
                    cur_precision_data = {};

                    // in total there are 17*len(jittering_levels) number of datasets for each jittering level
                    var num_datasets = 17*result_jittering_levels.length;

                    // 17 different results (as shown in 17 temporary containers
                    for (let a = 0; a < 17; a++) {
                        // when normal and shift-invariant, we have jittering levels
                        if (k == 0 || k == 1) {
                            for (let b = 0; b < result_jittering_levels.length; b++) {
                                var cur_jitter_level = result_jittering_levels[b];

                                // 1st len(result_jittering_levels) datasets are pred_id sorted by conf
                                if (a == 0) {
                                    // each image has 54 result files for each model
                                    cur_pred_id_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 2nd len(result_jittering_levels) datasets are pred_id sorted by iou
                                else if (a == 1) {
                                    cur_pred_id_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 3rd len(result_jittering_levels) datasets are pred_id sorted by conf*iou
                                if (a == 2) {
                                    // each image has 54 result files for each model
                                    cur_pred_id_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 4th len(result_jittering_levels) datasets are pred_id sorted by conf*iou*pr
                                else if (a == 3) {
                                    cur_pred_id_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }


                                // 5th len(result_jittering_levels) datasets are pred_bb sorted by conf
                                else if (a == 4) {
                                    cur_pred_bb_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 6th len(result_jittering_levels) datasets are pred_bb sorted by iou
                                else if (a == 5) {
                                    cur_pred_bb_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 7th len(result_jittering_levels) datasets are pred_bb sorted by conf*iou
                                else if (a == 6) {
                                    cur_pred_bb_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 8th len(result_jittering_levels) datasets are pred_bb sorted by conf*iou*pr
                                else if (a == 7) {
                                    cur_pred_bb_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }


                                // 9th len(result_jittering_levels) datasets are confidence sorted by conf
                                else if (a == 8) {
                                    cur_confidence_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 10th len(result_jittering_levels) datasets are confidence sorted by iou
                                else if (a == 9) {
                                    cur_confidence_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 11th len(result_jittering_levels) datasets are confidence sorted by conf*iou
                                else if (a == 10) {
                                    cur_confidence_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 12th len(result_jittering_levels) datasets are confidence sorted by conf*iou*pr
                                else if (a == 11) {
                                    cur_confidence_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }


                                // 13th len(result_jittering_levels) datasets are iou sorted by conf
                                else if (a == 12) {
                                    cur_iou_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 14th len(result_jittering_levels) datasets are iou sorted by iou
                                else if (a == 13) {
                                    cur_iou_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 15th len(result_jittering_levels) datasets are iou sorted by conf*iou
                                else if (a == 14) {
                                    cur_iou_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                                // 16th len(result_jittering_levels) datasets are iou sorted by conf*iou*pr
                                else if (a == 15) {
                                    cur_iou_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }


                                // ninth len(result_jittering_levels) datasets are precision
                                else if (a == 16) {
                                    cur_precision_data[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*num_datasets+a*result_jittering_levels.length+b];
                                }
                            }
                        }
                        // pretrained model results (no jittering)
                        else if (k == 2) {
                            var cur_jitter_level = 'all';

                            // 1st len(result_jittering_levels) datasets are pred_id sorted by conf
                            if (a == 0) {
                                cur_pred_id_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 2nd len(result_jittering_levels) datasets are pred_id sorted by iou
                            else if (a == 1) {
                                cur_pred_id_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 3rd len(result_jittering_levels) datasets are pred_id sorted by conf*iou
                            else if (a == 2) {
                                cur_pred_id_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 4th len(result_jittering_levels) datasets are pred_id sorted by conf*iou*pr
                            else if (a == 3) {
                                cur_pred_id_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }


                            // 5th len(result_jittering_levels) datasets are pred_bb sorted by conf
                            else if (a == 4) {
                                cur_pred_bb_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 6th len(result_jittering_levels) datasets are pred_bb sorted by iou
                            else if (a == 5) {
                                cur_pred_bb_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 7th len(result_jittering_levels) datasets are pred_bb sorted by conf*iou
                            else if (a == 6) {
                                cur_pred_bb_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 8th len(result_jittering_levels) datasets are pred_bb sorted by conf*iou*pr
                            else if (a == 7) {
                                cur_pred_bb_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }


                            // 9th len(result_jittering_levels) datasets are confidence sorted by conf
                            else if (a == 8) {
                                cur_confidence_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 10th len(result_jittering_levels) datasets are confidence sorted by iou
                            else if (a == 9) {
                                cur_confidence_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 11th len(result_jittering_levels) datasets are confidence sorted by conf*iou
                            else if (a == 10) {
                                cur_confidence_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 12th len(result_jittering_levels) datasets are confidence sorted by conf*iou*pr
                            else if (a == 11) {
                                cur_confidence_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }


                            // 13th len(result_jittering_levels) datasets are iou sorted by conf
                            else if (a == 12) {
                                cur_iou_data_sorted_by_conf[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 14th len(result_jittering_levels) datasets are iou sorted by iou
                            else if (a == 13) {
                                cur_iou_data_sorted_by_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 15th len(result_jittering_levels) datasets are iou sorted by conf
                            else if (a == 14) {
                                cur_iou_data_sorted_by_conf_iou[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                            // 16th len(result_jittering_levels) datasets are iou sorted by iou
                            else if (a == 15) {
                                cur_iou_data_sorted_by_conf_iou_pr[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }


                            // ninth len(result_jittering_levels) datasets are precision
                            else if (a == 16) {
                                cur_precision_data[cur_jitter_level] = all_result_data[k*(num_datasets*desired_classes.length)+i*17+a];
                            }
                        }
                    }

                    if (k == 0) {
                        normal_pred_id_data_sorted_by_conf.push(cur_pred_id_data_sorted_by_conf);
                        normal_pred_id_data_sorted_by_iou.push(cur_pred_id_data_sorted_by_iou);
                        normal_pred_id_data_sorted_by_conf_iou.push(cur_pred_id_data_sorted_by_conf_iou);
                        normal_pred_id_data_sorted_by_conf_iou_pr.push(cur_pred_id_data_sorted_by_conf_iou_pr);

                        normal_pred_bb_data_sorted_by_conf.push(cur_pred_bb_data_sorted_by_conf);
                        normal_pred_bb_data_sorted_by_iou.push(cur_pred_bb_data_sorted_by_iou);
                        normal_pred_bb_data_sorted_by_conf_iou.push(cur_pred_bb_data_sorted_by_conf_iou);
                        normal_pred_bb_data_sorted_by_conf_iou_pr.push(cur_pred_bb_data_sorted_by_conf_iou_pr);

                        normal_confidence_data_sorted_by_conf.push(cur_confidence_data_sorted_by_conf);
                        normal_confidence_data_sorted_by_iou.push(cur_confidence_data_sorted_by_iou);
                        normal_confidence_data_sorted_by_conf_iou.push(cur_confidence_data_sorted_by_conf_iou);
                        normal_confidence_data_sorted_by_conf_iou_pr.push(cur_confidence_data_sorted_by_conf_iou_pr);

                        normal_iou_data_sorted_by_conf.push(cur_iou_data_sorted_by_conf);
                        normal_iou_data_sorted_by_iou.push(cur_iou_data_sorted_by_iou);
                        normal_iou_data_sorted_by_conf_iou.push(cur_iou_data_sorted_by_conf_iou);
                        normal_iou_data_sorted_by_conf_iou_pr.push(cur_iou_data_sorted_by_conf_iou_pr);

                        normal_precision_data.push(cur_precision_data);
                    }
                    else if (k == 1) {
                        si_pred_id_data_sorted_by_conf.push(cur_pred_id_data_sorted_by_conf);
                        si_pred_id_data_sorted_by_iou.push(cur_pred_id_data_sorted_by_iou);
                        si_pred_id_data_sorted_by_conf_iou.push(cur_pred_id_data_sorted_by_conf_iou);
                        si_pred_id_data_sorted_by_conf_iou_pr.push(cur_pred_id_data_sorted_by_conf_iou_pr);

                        si_pred_bb_data_sorted_by_conf.push(cur_pred_bb_data_sorted_by_conf);
                        si_pred_bb_data_sorted_by_iou.push(cur_pred_bb_data_sorted_by_iou);
                        si_pred_bb_data_sorted_by_conf_iou.push(cur_pred_bb_data_sorted_by_conf_iou);
                        si_pred_bb_data_sorted_by_conf_iou_pr.push(cur_pred_bb_data_sorted_by_conf_iou_pr);

                        si_confidence_data_sorted_by_conf.push(cur_confidence_data_sorted_by_conf);
                        si_confidence_data_sorted_by_iou.push(cur_confidence_data_sorted_by_iou);
                        si_confidence_data_sorted_by_conf_iou.push(cur_confidence_data_sorted_by_conf_iou);
                        si_confidence_data_sorted_by_conf_iou_pr.push(cur_confidence_data_sorted_by_conf_iou_pr);

                        si_iou_data_sorted_by_conf.push(cur_iou_data_sorted_by_conf);
                        si_iou_data_sorted_by_iou.push(cur_iou_data_sorted_by_iou);
                        si_iou_data_sorted_by_conf_iou.push(cur_iou_data_sorted_by_conf_iou);
                        si_iou_data_sorted_by_conf_iou_pr.push(cur_iou_data_sorted_by_conf_iou_pr);

                        si_precision_data.push(cur_precision_data);
                    }
                    else if (k == 2) {
                        pt_pred_id_data_sorted_by_conf.push(cur_pred_id_data_sorted_by_conf);
                        pt_pred_id_data_sorted_by_iou.push(cur_pred_id_data_sorted_by_iou);
                        pt_pred_id_data_sorted_by_conf_iou.push(cur_pred_id_data_sorted_by_conf_iou);
                        pt_pred_id_data_sorted_by_conf_iou_pr.push(cur_pred_id_data_sorted_by_conf_iou_pr);

                        pt_pred_bb_data_sorted_by_conf.push(cur_pred_bb_data_sorted_by_conf);
                        pt_pred_bb_data_sorted_by_iou.push(cur_pred_bb_data_sorted_by_iou);
                        pt_pred_bb_data_sorted_by_conf_iou.push(cur_pred_bb_data_sorted_by_conf_iou);
                        pt_pred_bb_data_sorted_by_conf_iou_pr.push(cur_pred_bb_data_sorted_by_conf_iou_pr);

                        pt_confidence_data_sorted_by_conf.push(cur_confidence_data_sorted_by_conf);
                        pt_confidence_data_sorted_by_iou.push(cur_confidence_data_sorted_by_iou);
                        pt_confidence_data_sorted_by_conf_iou.push(cur_confidence_data_sorted_by_conf_iou);
                        pt_confidence_data_sorted_by_conf_iou_pr.push(cur_confidence_data_sorted_by_conf_iou_pr);

                        pt_iou_data_sorted_by_conf.push(cur_iou_data_sorted_by_conf);
                        pt_iou_data_sorted_by_iou.push(cur_iou_data_sorted_by_iou);
                        pt_iou_data_sorted_by_conf_iou.push(cur_iou_data_sorted_by_conf_iou);
                        pt_iou_data_sorted_by_conf_iou_pr.push(cur_iou_data_sorted_by_conf_iou_pr);

                        pt_precision_data.push(cur_precision_data);
                    }
                }
            }

            console.log('Model outputs data from ', desired_classes, 'has been loaded successfully.')

                // plot_heatmap_canvas(parseInt(cur_image_index),
                //                     cur_jittering_level,
                //                     cur_sort_method,
                //                     cur_plot_quantity,
                //                     target_id_data,
                //                     target_bb_data,
                //                     normal_pred_id_data_sorted_by_conf,
                //                     normal_pred_bb_data_sorted_by_conf,
                //                     normal_confidence_data_sorted_by_conf,
                //                     normal_iou_data_sorted_by_conf,
                //                     normal_pred_id_data_sorted_by_iou,
                //                     normal_pred_bb_data_sorted_by_iou,
                //                     normal_confidence_data_sorted_by_iou,
                //                     normal_iou_data_sorted_by_iou,
                //                     normal_precision_data,
                //                     data_container,
                //                     normal_tooltip,
                //                     cutout_pos_data);
        }
    });
}
