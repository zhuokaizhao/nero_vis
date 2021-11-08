// This script communicates between control inputs and re-plotting, etc


// controllers inputs
var vis_mode_button = d3.selectAll("input[name='vis_mode']");
var si_method_button = d3.selectAll("input[name='si_method']");
var object_class_button = d3.selectAll("input[name='object_class']");
var sort_method_button = d3.selectAll("input[name='sort_method']");
var plot_test_quantity_button = d3.selectAll("input[name='plot_test_quantity']");
var plot_single_test_quantity_button = d3.selectAll("input[name='plot_single_test_quantity']");
// var jittering_slider = d3.select('#jittering');
var jittering_button = d3.selectAll("input[name='jittering']");

// vis mode input (general or single-test)
vis_mode_button.on('change', function() {
    cur_vis_mode = this.value;
    if (cur_vis_mode == 'test') {
        cur_plot_quantity = 'AP';
        document.getElementById('shift_equivariant_method').style.display = 'none';
        document.getElementById('test_plot_quantities').style.display = 'block';
        document.getElementById('single_test_plot_quantities').style.display = 'none';

        // turn off the single-test div's
        normal_svg.selectAll('*').remove();
        si_svg.selectAll('*').remove();
        tooltip_svg.selectAll('*').remove();

        // re-plot test
        plot_test_heatmap('normal', normal_test_data, normal_svg);
        plot_test_heatmap('si', si_test_data, si_svg);
        plot_test_heatmap('pt', pt_test_data, pt_svg);
    }
    else if (cur_vis_mode == 'single_test') {
        // update plot quantity for single-test
        cur_plot_quantity = 0;

        document.getElementById('shift_equivariant_method').style.display = 'block';
        document.getElementById('test_plot_quantities').style.display = 'none';
        document.getElementById('single_test_plot_quantities').style.display = 'block';

        // turn off the test div's
        normal_svg.selectAll('*').remove();
        si_svg.selectAll('*').remove();
        pt_svg.selectAll('*').remove();
        // d3.select('pt_svg').remove();

        // tooltip_svg.selectAll('*').remove();

        // show the original image in d3 svg
        var tooltip_image = tooltip_svg.selectAll("image").data([0]);
        // assign image link by image index
        image_path = data_dir.concat('images/', cur_object_class, '.jpg');
        tooltip_image.enter()
                        .append("svg:image")
                        .attr("xlink:href", image_path)
                        .style("opacity", 0.2);

        // plot the single-test
        // plot for normal model
        plot_single_test_heatmap('normal',
                                target_id_data,
                                target_bb_data,
                                normal_pred_id_data_sorted_by_conf,
                                normal_pred_bb_data_sorted_by_conf,
                                normal_confidence_data_sorted_by_conf,
                                normal_iou_data_sorted_by_conf,
                                normal_pred_id_data_sorted_by_iou,
                                normal_pred_bb_data_sorted_by_iou,
                                normal_confidence_data_sorted_by_iou,
                                normal_iou_data_sorted_by_iou,
                                normal_pred_id_data_sorted_by_conf_iou,
                                normal_pred_bb_data_sorted_by_conf_iou,
                                normal_confidence_data_sorted_by_conf_iou,
                                normal_iou_data_sorted_by_conf_iou,
                                normal_pred_id_data_sorted_by_conf_iou_pr,
                                normal_pred_bb_data_sorted_by_conf_iou_pr,
                                normal_confidence_data_sorted_by_conf_iou_pr,
                                normal_iou_data_sorted_by_conf_iou_pr,
                                normal_precision_data,
                                cutout_pos_data,
                                normal_svg);

        // plot for si svg
        // shift-invariant model (chosen from pretrained or si)
        if (cur_si_method == 'pt') {
            plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    pt_pred_id_data_sorted_by_conf,
                                    pt_pred_bb_data_sorted_by_conf,
                                    pt_confidence_data_sorted_by_conf,
                                    pt_iou_data_sorted_by_conf,
                                    pt_pred_id_data_sorted_by_iou,
                                    pt_pred_bb_data_sorted_by_iou,
                                    pt_confidence_data_sorted_by_iou,
                                    pt_iou_data_sorted_by_iou,
                                    pt_pred_id_data_sorted_by_conf_iou,
                                    pt_pred_bb_data_sorted_by_conf_iou,
                                    pt_confidence_data_sorted_by_conf_iou,
                                    pt_iou_data_sorted_by_conf_iou,
                                    pt_pred_id_data_sorted_by_conf_iou_pr,
                                    pt_pred_bb_data_sorted_by_conf_iou_pr,
                                    pt_confidence_data_sorted_by_conf_iou_pr,
                                    pt_iou_data_sorted_by_conf_iou_pr,
                                    pt_precision_data,
                                    cutout_pos_data,
                                    si_svg);
        }
        else if (cur_si_method == 'si') {
            plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    si_pred_id_data_sorted_by_conf,
                                    si_pred_bb_data_sorted_by_conf,
                                    si_confidence_data_sorted_by_conf,
                                    si_iou_data_sorted_by_conf,
                                    si_pred_id_data_sorted_by_iou,
                                    si_pred_bb_data_sorted_by_iou,
                                    si_confidence_data_sorted_by_iou,
                                    si_iou_data_sorted_by_iou,
                                    si_pred_id_data_sorted_by_conf_iou,
                                    si_pred_bb_data_sorted_by_conf_iou,
                                    si_confidence_data_sorted_by_conf_iou,
                                    si_iou_data_sorted_by_conf_iou,
                                    si_pred_id_data_sorted_by_conf_iou_pr,
                                    si_pred_bb_data_sorted_by_conf_iou_pr,
                                    si_confidence_data_sorted_by_conf_iou_pr,
                                    si_iou_data_sorted_by_conf_iou_pr,
                                    si_precision_data,
                                    cutout_pos_data,
                                    si_svg);
        }
    }
})


// si method input (si or pretrained)
si_method_button.on('change', function() {
    cur_si_method = this.value;
    // console.log(cur_si_method)
    // d3.select("#si_title").remove();
    // si_svg.append("text")
    //         .attr("x", title_x1)
    //         .attr("y", title_y1)
    //         .attr("text-anchor", "center")
    //         .style("font-size", "22px")
    //         .text("Shift-equavariant model (" + cur_si_method + ")").attr('id', 'si_title');;
    // shift-invariant model (chosen from pretrained or si)
    if (cur_si_method == 'pt') {
        plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    pt_pred_id_data_sorted_by_conf,
                                    pt_pred_bb_data_sorted_by_conf,
                                    pt_confidence_data_sorted_by_conf,
                                    pt_iou_data_sorted_by_conf,
                                    pt_pred_id_data_sorted_by_iou,
                                    pt_pred_bb_data_sorted_by_iou,
                                    pt_confidence_data_sorted_by_iou,
                                    pt_iou_data_sorted_by_iou,
                                    pt_pred_id_data_sorted_by_conf_iou,
                                    pt_pred_bb_data_sorted_by_conf_iou,
                                    pt_confidence_data_sorted_by_conf_iou,
                                    pt_iou_data_sorted_by_conf_iou,
                                    pt_pred_id_data_sorted_by_conf_iou_pr,
                                    pt_pred_bb_data_sorted_by_conf_iou_pr,
                                    pt_confidence_data_sorted_by_conf_iou_pr,
                                    pt_iou_data_sorted_by_conf_iou_pr,
                                    pt_precision_data,
                                    cutout_pos_data,
                                    si_svg);
    }
    else if (cur_si_method == 'si') {
        plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    si_pred_id_data_sorted_by_conf,
                                    si_pred_bb_data_sorted_by_conf,
                                    si_confidence_data_sorted_by_conf,
                                    si_iou_data_sorted_by_conf,
                                    si_pred_id_data_sorted_by_iou,
                                    si_pred_bb_data_sorted_by_iou,
                                    si_confidence_data_sorted_by_iou,
                                    si_iou_data_sorted_by_iou,
                                    si_pred_id_data_sorted_by_conf_iou,
                                    si_pred_bb_data_sorted_by_conf_iou,
                                    si_confidence_data_sorted_by_conf_iou,
                                    si_iou_data_sorted_by_conf_iou,
                                    si_pred_id_data_sorted_by_conf_iou_pr,
                                    si_pred_bb_data_sorted_by_conf_iou_pr,
                                    si_confidence_data_sorted_by_conf_iou_pr,
                                    si_iou_data_sorted_by_conf_iou_pr,
                                    si_precision_data,
                                    cutout_pos_data,
                                    si_svg);
    }
});


// when object class changes (all five desire classes)
object_class_button.on('change', function() {
    cur_object_class = this.value;

    if (cur_vis_mode == 'test') {
        plot_test_heatmap('normal', normal_test_data, normal_svg);
        plot_test_heatmap('si', si_test_data, si_svg);
        plot_test_heatmap('pt', pt_test_data, pt_svg);
    }
    else if (cur_vis_mode == 'single_test') {
        // get space for tooltip
        // pt_svg.selectAll('*').remove();
        tooltip_svg.selectAll('*').remove();

        // show the original image in d3 svg
        var tooltip_image = tooltip_svg.selectAll("image").data([0]);
        // assign image link by image index
        image_path = data_dir.concat('images/', cur_object_class, '.jpg');
        tooltip_image.enter()
                        .append("svg:image")
                        .attr("xlink:href", image_path)
                        .style("opacity", 0.2);

        // plot for normal model
        plot_single_test_heatmap('normal',
                                target_id_data,
                                target_bb_data,
                                normal_pred_id_data_sorted_by_conf,
                                normal_pred_bb_data_sorted_by_conf,
                                normal_confidence_data_sorted_by_conf,
                                normal_iou_data_sorted_by_conf,
                                normal_pred_id_data_sorted_by_iou,
                                normal_pred_bb_data_sorted_by_iou,
                                normal_confidence_data_sorted_by_iou,
                                normal_iou_data_sorted_by_iou,
                                normal_pred_id_data_sorted_by_conf_iou,
                                normal_pred_bb_data_sorted_by_conf_iou,
                                normal_confidence_data_sorted_by_conf_iou,
                                normal_iou_data_sorted_by_conf_iou,
                                normal_pred_id_data_sorted_by_conf_iou_pr,
                                normal_pred_bb_data_sorted_by_conf_iou_pr,
                                normal_confidence_data_sorted_by_conf_iou_pr,
                                normal_iou_data_sorted_by_conf_iou_pr,
                                normal_precision_data,
                                cutout_pos_data,
                                normal_svg);

        // plot for si svg
        // shift-invariant model (chosen from pretrained or si)
        if (cur_si_method == 'pt') {
            plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    pt_pred_id_data_sorted_by_conf,
                                    pt_pred_bb_data_sorted_by_conf,
                                    pt_confidence_data_sorted_by_conf,
                                    pt_iou_data_sorted_by_conf,
                                    pt_pred_id_data_sorted_by_iou,
                                    pt_pred_bb_data_sorted_by_iou,
                                    pt_confidence_data_sorted_by_iou,
                                    pt_iou_data_sorted_by_iou,
                                    pt_pred_id_data_sorted_by_conf_iou,
                                    pt_pred_bb_data_sorted_by_conf_iou,
                                    pt_confidence_data_sorted_by_conf_iou,
                                    pt_iou_data_sorted_by_conf_iou,
                                    pt_pred_id_data_sorted_by_conf_iou_pr,
                                    pt_pred_bb_data_sorted_by_conf_iou_pr,
                                    pt_confidence_data_sorted_by_conf_iou_pr,
                                    pt_iou_data_sorted_by_conf_iou_pr,
                                    pt_precision_data,
                                    cutout_pos_data,
                                    si_svg);
        }
        else if (cur_si_method == 'si') {
            plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    si_pred_id_data_sorted_by_conf,
                                    si_pred_bb_data_sorted_by_conf,
                                    si_confidence_data_sorted_by_conf,
                                    si_iou_data_sorted_by_conf,
                                    si_pred_id_data_sorted_by_iou,
                                    si_pred_bb_data_sorted_by_iou,
                                    si_confidence_data_sorted_by_iou,
                                    si_iou_data_sorted_by_iou,
                                    si_pred_id_data_sorted_by_conf_iou,
                                    si_pred_bb_data_sorted_by_conf_iou,
                                    si_confidence_data_sorted_by_conf_iou,
                                    si_iou_data_sorted_by_conf_iou,
                                    si_pred_id_data_sorted_by_conf_iou_pr,
                                    si_pred_bb_data_sorted_by_conf_iou_pr,
                                    si_confidence_data_sorted_by_conf_iou_pr,
                                    si_iou_data_sorted_by_conf_iou_pr,
                                    si_precision_data,
                                    cutout_pos_data,
                                    si_svg);
        }
    }
});


// when test plotted quantity is changed
plot_test_quantity_button.on('change', function() {
    cur_plot_quantity = this.value;
    // plot for normal model
    plot_test_heatmap('normal', normal_test_data, normal_svg);
    plot_test_heatmap('si', si_test_data, si_svg);
    plot_test_heatmap('pt', pt_test_data, pt_svg);
})

// when single-test plotted quantity is changed
plot_single_test_quantity_button.on('change', function() {
    var plot_button_input = this.value;
    if (plot_button_input == "conf") {
        cur_plot_quantity = 0;
    }
    else if (plot_button_input == 'iou') {
        cur_plot_quantity = 1;
    }
    else if (plot_button_input == 'cf_iou') {
        cur_plot_quantity = 2;
    }
    else if (plot_button_input == 'cf_iou_pr') {
        cur_plot_quantity = 3;
    }

    // plot for normal model
    plot_single_test_heatmap('normal',
                                target_id_data,
                                target_bb_data,
                                normal_pred_id_data_sorted_by_conf,
                                normal_pred_bb_data_sorted_by_conf,
                                normal_confidence_data_sorted_by_conf,
                                normal_iou_data_sorted_by_conf,
                                normal_pred_id_data_sorted_by_iou,
                                normal_pred_bb_data_sorted_by_iou,
                                normal_confidence_data_sorted_by_iou,
                                normal_iou_data_sorted_by_iou,
                                normal_pred_id_data_sorted_by_conf_iou,
                                normal_pred_bb_data_sorted_by_conf_iou,
                                normal_confidence_data_sorted_by_conf_iou,
                                normal_iou_data_sorted_by_conf_iou,
                                normal_pred_id_data_sorted_by_conf_iou_pr,
                                normal_pred_bb_data_sorted_by_conf_iou_pr,
                                normal_confidence_data_sorted_by_conf_iou_pr,
                                normal_iou_data_sorted_by_conf_iou_pr,
                                normal_precision_data,
                                cutout_pos_data,
                                normal_svg);

    // plot for si model
    // shift-invariant model (chosen from pretrained or si)
    if (cur_si_method == 'pt') {
        plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    pt_pred_id_data_sorted_by_conf,
                                    pt_pred_bb_data_sorted_by_conf,
                                    pt_confidence_data_sorted_by_conf,
                                    pt_iou_data_sorted_by_conf,
                                    pt_pred_id_data_sorted_by_iou,
                                    pt_pred_bb_data_sorted_by_iou,
                                    pt_confidence_data_sorted_by_iou,
                                    pt_iou_data_sorted_by_iou,
                                    pt_pred_id_data_sorted_by_conf_iou,
                                    pt_pred_bb_data_sorted_by_conf_iou,
                                    pt_confidence_data_sorted_by_conf_iou,
                                    pt_iou_data_sorted_by_conf_iou,
                                    pt_pred_id_data_sorted_by_conf_iou_pr,
                                    pt_pred_bb_data_sorted_by_conf_iou_pr,
                                    pt_confidence_data_sorted_by_conf_iou_pr,
                                    pt_iou_data_sorted_by_conf_iou_pr,
                                    pt_precision_data,
                                    cutout_pos_data,
                                    si_svg);
    }
    else if (cur_si_method == 'si') {
        plot_single_test_heatmap(cur_si_method,
                                    target_id_data,
                                    target_bb_data,
                                    si_pred_id_data_sorted_by_conf,
                                    si_pred_bb_data_sorted_by_conf,
                                    si_confidence_data_sorted_by_conf,
                                    si_iou_data_sorted_by_conf,
                                    si_pred_id_data_sorted_by_iou,
                                    si_pred_bb_data_sorted_by_iou,
                                    si_confidence_data_sorted_by_iou,
                                    si_iou_data_sorted_by_iou,
                                    si_pred_id_data_sorted_by_conf_iou,
                                    si_pred_bb_data_sorted_by_conf_iou,
                                    si_confidence_data_sorted_by_conf_iou,
                                    si_iou_data_sorted_by_conf_iou,
                                    si_pred_id_data_sorted_by_conf_iou_pr,
                                    si_pred_bb_data_sorted_by_conf_iou_pr,
                                    si_confidence_data_sorted_by_conf_iou_pr,
                                    si_iou_data_sorted_by_conf_iou_pr,
                                    si_precision_data,
                                    cutout_pos_data,
                                    si_svg);
    }
});

// when jittering level is changed
jittering_button.on('change', function() {
    cur_jittering_level = this.value;

    if (cur_vis_mode == 'test') {
        // plot for normal model
        plot_test_heatmap('normal', normal_test_data, normal_svg);
        plot_test_heatmap('si', si_test_data, si_svg);
        plot_test_heatmap('pt', pt_test_data, pt_svg);
    }
    else if (cur_vis_mode == 'single_test') {
        plot_single_test_heatmap('normal',
                                target_id_data,
                                target_bb_data,
                                normal_pred_id_data_sorted_by_conf,
                                normal_pred_bb_data_sorted_by_conf,
                                normal_confidence_data_sorted_by_conf,
                                normal_iou_data_sorted_by_conf,
                                normal_pred_id_data_sorted_by_iou,
                                normal_pred_bb_data_sorted_by_iou,
                                normal_confidence_data_sorted_by_iou,
                                normal_iou_data_sorted_by_iou,
                                normal_pred_id_data_sorted_by_conf_iou,
                                normal_pred_bb_data_sorted_by_conf_iou,
                                normal_confidence_data_sorted_by_conf_iou,
                                normal_iou_data_sorted_by_conf_iou,
                                normal_pred_id_data_sorted_by_conf_iou_pr,
                                normal_pred_bb_data_sorted_by_conf_iou_pr,
                                normal_confidence_data_sorted_by_conf_iou_pr,
                                normal_iou_data_sorted_by_conf_iou_pr,
                                normal_precision_data,
                                cutout_pos_data,
                                normal_svg);

        // jittering level change does not affect pretrained model
        if (cur_si_method == 'si') {
            plot_single_test_heatmap(cur_si_method,
                                        target_id_data,
                                        target_bb_data,
                                        si_pred_id_data_sorted_by_conf,
                                        si_pred_bb_data_sorted_by_conf,
                                        si_confidence_data_sorted_by_conf,
                                        si_iou_data_sorted_by_conf,
                                        si_pred_id_data_sorted_by_iou,
                                        si_pred_bb_data_sorted_by_iou,
                                        si_confidence_data_sorted_by_iou,
                                        si_iou_data_sorted_by_iou,
                                        si_pred_id_data_sorted_by_conf_iou,
                                        si_pred_bb_data_sorted_by_conf_iou,
                                        si_confidence_data_sorted_by_conf_iou,
                                        si_iou_data_sorted_by_conf_iou,
                                        si_pred_id_data_sorted_by_conf_iou_pr,
                                        si_pred_bb_data_sorted_by_conf_iou_pr,
                                        si_confidence_data_sorted_by_conf_iou_pr,
                                        si_iou_data_sorted_by_conf_iou_pr,
                                        si_precision_data,
                                        cutout_pos_data,
                                        si_svg);
        }
    }
});
