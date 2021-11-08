// The script holds all the ploting functions

// plot heatmaps for test cases
function plot_test_heatmap(mode, test_results, test_svg) {

    // Build color scale
    var colormap_test = d3.scaleSequential()
                        .domain([0, 0.5])
                        .interpolator(d3.interpolateViridis);

    if (mode == 'normal') {
        // title of the heatmap
        var title = 'Non-shift-equivariant model';
        var offset_x = 0;
    }
    else if (mode == 'si') {
        // title of the heatmap
        var title = 'Shift-equivariant model';
        var offset_x = 25;
    }
    else if (mode == 'pt') {
        var title = 'Pre-trained model';
        var offset_x = 35;
    }

    // Add title to graph
    test_svg.append("text")
            .attr("x", title_x+offset_x)
            .attr("y", title_y)
            .attr("text-anchor", "center")
            .style("font-size", "22px")
            .text(title);


    // Add subtitle to graph
    // var subtitle = 'Plotting ' + cur_plot_quantity;
    // test_svg.append("text")
    //         .attr("x", subtitle_x)
    //         .attr("y", subtitle_y)
    //         .attr("text-anchor", "center")
    //         .style("font-size", "14px")
    //         .style("fill", "grey")
    //         .style("max-width", 400)
    //         .text(subtitle).attr('id', mode.concat('_subtitle_test'));

    // determine the data by jittering levels and plot quantity
    if (mode == 'pt') {
        cur_plot_data = test_results[cur_plot_quantity]['all'];
    }
    else {
        cur_plot_data = test_results[cur_plot_quantity][cur_jittering_level];
    }

    // build for unequal interval
    translations = [-1, -2, -3, -4, -6, -8, -10, -13, -17, -22, -29, -37, -48, -63].reverse().concat([0, 1, 2, 3, 4, 6, 8, 10, 13, 17, 22, 29, 37, 48, 63]);
    all_gaps = [16, 14, 8, 8, 6, 4, 4, 2, 2, 2, 1, 1, 1, 1].concat(1).concat([16, 14, 8, 8, 6, 4, 4, 2, 2, 2, 1, 1, 1, 1].reverse());
    // ticks = [-4, -10, -22, -37, -48, -63].reverse().concat([0, 4, 10, 22, 37, 48, 63]);

    // cq is used to determine color
    var cur_cq_data = [];
    // plotting confidence
    if (cur_object_class == 'car') {
        for (var i = 0; i < cur_plot_data.length; i++) {
            var row = parseInt(i / translations.length);
            var col = parseInt(i % translations.length);

            cur_cq_data.push({'x_tran': cur_plot_data[i].x_tran,
                            'y_tran': cur_plot_data[i].y_tran,
                            'cq': cur_plot_data[i].car,
                            'width': all_gaps[col],
                            'height': all_gaps[row]})
        }
    }
    else if (cur_object_class == 'bottle') {
        for (var i = 0; i < cur_plot_data.length; i++) {
            var row = parseInt(i / translations.length);
            var col = parseInt(i % translations.length);

            cur_cq_data.push({'x_tran': cur_plot_data[i].x_tran,
                                'y_tran': cur_plot_data[i].y_tran,
                                'cq': cur_plot_data[i].bottle,
                                'width': all_gaps[col],
                                'height': all_gaps[row]})
        }
    }
    else if (cur_object_class == 'cup') {
        for (var i = 0; i < cur_plot_data.length; i++) {
            var row = parseInt(i / translations.length);
            var col = parseInt(i % translations.length);

            cur_cq_data.push({'x_tran': cur_plot_data[i].x_tran,
                                'y_tran': cur_plot_data[i].y_tran,
                                'cq': cur_plot_data[i].cup,
                                'width': all_gaps[col],
                                'height': all_gaps[row]})
        }
    }
    else if (cur_object_class == 'chair') {
        for (var i = 0; i < cur_plot_data.length; i++) {
            var row = parseInt(i / translations.length);
            var col = parseInt(i % translations.length);

            cur_cq_data.push({'x_tran': cur_plot_data[i].x_tran,
                                'y_tran': cur_plot_data[i].y_tran,
                                'cq': cur_plot_data[i].chair,
                                'width': all_gaps[col],
                                'height': all_gaps[row]})
        }
    }
    else if (cur_object_class == 'book') {
        for (var i = 0; i < cur_plot_data.length; i++) {
            var row = parseInt(i / translations.length);
            var col = parseInt(i % translations.length);

            cur_cq_data.push({'x_tran': cur_plot_data[i].x_tran,
                                'y_tran': cur_plot_data[i].y_tran,
                                'cq': cur_plot_data[i].book,
                                'width': all_gaps[col],
                                'height': all_gaps[row]})
        }
    }

    // Labels of row (y_translations) and columns (x_translations)
    var x_translations = d3.map(cur_cq_data, function(d){return d.x_tran;}).keys();
    var y_translations = d3.map(cur_cq_data, function(d){return d.y_tran;}).keys();

    // Build X scales and axis:
    var x_scale = d3.scaleLinear()
                    .range([ 0, width ])
                    .domain([-63, 63]);
    var x_axis = d3.axisBottom(x_scale)
                    .tickValues(x_translations)
                    .tickSize(0);

    // Build Y scales and axis:
    var y_scale = d3.scaleLinear()
                    .range([ 0, height ])
                    .domain([-63, 63]);
    var y_axis = d3.axisLeft(y_scale)
                    .tickValues(y_translations)
                    .tickSize(0);
    test_svg.append("g")
            .style("font-size", 15)
            .attr("transform", "translate(0," + y_scale(64) + ")")
            // .call(x_axis)
            .select(".domain").remove()
    test_svg.append("g")
            .style("font-size", 15)
            .attr("transform", "translate(" + x_scale(-64) + ",0)")
            // .call(y_axis)
            .select(".domain").remove()

    // click event
    // var hover_turn_off = false;
    // var mouseclick = function(d) {
    //     if (hover_turn_off == true) {
    //         hover_turn_off = false;
    //         d3.selectAll("#chosen_rect")
    //             .style("stroke", "none")
    //             .style("opacity", 1)
    //     }
    //     else {
    //         hover_turn_off = true;
    //         // make the selected rect to red
    //         d3.select(this)
    //             .style("stroke", "red")
    //             .style("opacity", 1)
    //             .attr('id', 'chosen_rect')
    //     }
    // }

    // // Three function that change the tooltip when user hover / move / leave a cell
    // var mouseover = function(d) {
    //     if (hover_turn_off == false) {
    //         tooltip.style("opacity", 1)
    //         d3.select(this)
    //             .style("stroke", "black")
    //             .style("opacity", 1)
    //     }
    // }
    // var mousemove = function(d) {
    //     if (hover_turn_off == false) {

    //         var hover_html = [];
    //         hover_html.push('x translation: ' + d.x_tran +
    //                         ', y translation: ' + d.y_tran + ', ' + '</br>')
    //         tooltip.html(hover_html)
    //                 .style("left", (d3.mouse(this)[0]+70) + "px")
    //                 .style("top", (d3.mouse(this)[1]) + "px")
    //     }
    // }
    // var mouseleave = function(d) {
    //     if (hover_turn_off == false) {
    //         tooltip.style("opacity", 0)
    //         d3.select(this)
    //             .style("stroke", "none")
    //             .style("opacity", 1)
    //     }
    // }

    // add the squares
    // heat map is created differently by input
    test_svg.selectAll('rect').remove();
    test_svg.selectAll('rect')
            .data(cur_cq_data, function(d) {return d.cq;})
            .enter()
            .append("rect")
            .attr("x", function(d) { return x_scale(d.x_tran-d.width/2) })
            .attr("y", function(d) { return y_scale(d.y_tran-d.height/2) })
            .attr("rx", 0)
            .attr("ry", 0)
            .attr("width", function (d) { return x_scale(d.width); } )
            .attr("height", function (d) { return y_scale(d.height); } )
            .style("fill", function(d) { return colormap_test(d.cq)} )
            .style("stroke-width", 2)
            .style("stroke", "none")
            .style("opacity", 1)
            // .on("click", mouseclick)
            // .on("mouseover", mouseover)
            // .on("mousemove", mousemove)
            // .on("mouseleave", mouseleave)
}


// plot heatmaps for single-test cases
function plot_single_test_heatmap(mode,
                                    target_id_data,
                                    target_bb_data,
                                    pred_id_data_sorted_by_conf,
                                    pred_bb_data_sorted_by_conf,
                                    confidence_data_sorted_by_conf,
                                    iou_data_sorted_by_conf,
                                    pred_id_data_sorted_by_iou,
                                    pred_bb_data_sorted_by_iou,
                                    confidence_data_sorted_by_iou,
                                    iou_data_sorted_by_iou,
                                    pred_id_data_sorted_by_conf_iou,
                                    pred_bb_data_sorted_by_conf_iou,
                                    confidence_data_sorted_by_conf_iou,
                                    iou_data_sorted_by_conf_iou,
                                    pred_id_data_sorted_by_conf_iou_pr,
                                    pred_bb_data_sorted_by_conf_iou_pr,
                                    confidence_data_sorted_by_conf_iou_pr,
                                    iou_data_sorted_by_conf_iou_pr,
                                    precision_data,
                                    cutout_pos_data,
                                    single_test_svg) {

    var colormap_single_test = d3.scaleSequential()
                                    .domain([0, 1])
                                    .interpolator(d3.interpolateViridis);

    if (mode == 'normal') {
        // title of the heatmap
        var title = 'Non-shift-equivariant model';
    }
    else if (mode == 'si' || mode == 'pretrained') {
        // title of the heatmap
        var title = 'Shift-equivariant model';
    }


    // Add title to graph
    single_test_svg.append("text")
                        .attr("x", title_x)
                        .attr("y", title_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "22px")
                        .text(title);


    // Add subtitle to graph
    // var subtitle = 'Plotting ' + cur_plot_quantity;
    // single_test_svg.append("text")
    //                     .attr("x", subtitle_x)
    //                     .attr("y", subtitle_y)
    //                     .attr("text-anchor", "center")
    //                     .style("font-size", "14px")
    //                     .style("fill", "grey")
    //                     .style("max-width", 400)
    //                     .text(subtitle).attr('id', mode.concat('_subtitle_single_test'));

    // pretrained model results does not have different jittering levels
    var jittering;
    if (mode == 'pt') {
        jittering = 'all';
    }
    else {
        jittering = cur_jittering_level;
    }

    // choose the set of data by input
    var cur_pred_id_data;
    var cur_pred_bb_data;
    var cur_confidence_data;
    var cur_iou_data;
    var cur_precision_data = precision_data[cur_plot_quantity][jittering];
    // console.log(image_index, confidence_data_sorted_by_conf[image_index]);
    // sorting is now the same as plot quantity
    // sorted by conf
    // console.log(pred_id_data_sorted_by_conf)
    var object_class_index;
    if (cur_object_class == desired_classes[0]) {
        object_class_index = 0
    }
    else if (cur_object_class == desired_classes[1]) {
        object_class_index = 1
    }
    else if (cur_object_class == desired_classes[2]) {
        object_class_index = 2
    }
    else if (cur_object_class == desired_classes[3]) {
        object_class_index = 3
    }
    else if (cur_object_class == desired_classes[4]) {
        object_class_index = 4
    }

    // sorted by conf
    if (cur_plot_quantity == 0) {
        cur_pred_id_data = pred_id_data_sorted_by_conf[object_class_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_conf[object_class_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_conf[object_class_index][jittering]
        cur_iou_data = iou_data_sorted_by_conf[object_class_index][jittering]
    // sorted by iou
    } else if (cur_plot_quantity == 1) {
        cur_pred_id_data = pred_id_data_sorted_by_iou[object_class_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_iou[object_class_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_iou[object_class_index][jittering]
        cur_iou_data = iou_data_sorted_by_iou[object_class_index][jittering]
    }
    // sorted by conf*iou
    else if (cur_plot_quantity == 2) {
        cur_pred_id_data = pred_id_data_sorted_by_conf_iou[object_class_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_conf_iou[object_class_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_conf_iou[object_class_index][jittering]
        cur_iou_data = iou_data_sorted_by_conf_iou[object_class_index][jittering]
    }
    // sorted by conf*iou*pr
    else if (cur_plot_quantity == 3) {
        cur_pred_id_data = pred_id_data_sorted_by_conf_iou_pr[object_class_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_conf_iou_pr[object_class_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_conf_iou_pr[object_class_index][jittering]
        cur_iou_data = iou_data_sorted_by_conf_iou_pr[object_class_index][jittering]
    }
    console.log(confidence_data_sorted_by_conf[object_class_index])
    console.log(cur_confidence_data)

    // determine the quantity that we are plotting, take three for later plotting purpose
    // cq is used to determine color
    var cur_cq_data = [];
    // plotting confidence
    if (cur_plot_quantity == 0) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'id': desired_classes[parseInt(cur_pred_id_data[i].id)],
                                'id1': desired_classes[parseInt(cur_pred_id_data[i].id1)],
                                'id2': desired_classes[parseInt(cur_pred_id_data[i].id2)],
                                'cq': cur_confidence_data[i].confidence,
                                'cq1': cur_confidence_data[i].confidence1,
                                'cq2': cur_confidence_data[i].confidence2,
                                'conf': cur_confidence_data[i].confidence,
                                'conf1': cur_confidence_data[i].confidence1,
                                'conf2': cur_confidence_data[i].confidence2,
                                'iou': cur_iou_data[i].iou,
                                'iou1': cur_iou_data[i].iou1,
                                'iou2': cur_iou_data[i].iou2,
                                'conf_iou': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou),
                                'conf_iou1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1),
                                'conf_iou2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2),
                                'conf_iou_pr': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision),
                                'conf_iou_pr1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1)*parseFloat(cur_precision_data[i].precision1),
                                'conf_iou_pr2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2)*parseFloat(cur_precision_data[i].precision2)})
        }
        // console.log('cur_confidence_data', cur_confidence_data);
        // console.log('cur_cq_data', cur_cq_data);
    }
    // plotting iou
    else if (cur_plot_quantity == 1) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'id': desired_classes[parseInt(cur_pred_id_data[i].id)],
                                'id1': desired_classes[parseInt(cur_pred_id_data[i].id1)],
                                'id2': desired_classes[parseInt(cur_pred_id_data[i].id2)],
                                'cq': cur_iou_data[i].iou,
                                'cq1': cur_iou_data[i].iou1,
                                'cq2': cur_iou_data[i].iou2,
                                'conf': cur_confidence_data[i].confidence,
                                'conf1': cur_confidence_data[i].confidence1,
                                'conf2': cur_confidence_data[i].confidence2,
                                'iou': cur_iou_data[i].iou,
                                'iou1': cur_iou_data[i].iou1,
                                'iou2': cur_iou_data[i].iou2,
                                'conf_iou': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou),
                                'conf_iou1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1),
                                'conf_iou2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2),
                                'conf_iou_pr': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision),
                                'conf_iou_pr1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1)*parseFloat(cur_precision_data[i].precision1),
                                'conf_iou_pr2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2)*parseFloat(cur_precision_data[i].precision2)})
        }
    }
    // plotting confidence*IOU
    else if (cur_plot_quantity == 2) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'id': desired_classes[parseInt(cur_pred_id_data[i].id)],
                                'id1': desired_classes[parseInt(cur_pred_id_data[i].id1)],
                                'id2': desired_classes[parseInt(cur_pred_id_data[i].id2)],
                                'cq': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou),
                                'cq1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1),
                                'cq2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2),
                                'conf': cur_confidence_data[i].confidence,
                                'conf1': cur_confidence_data[i].confidence1,
                                'conf2': cur_confidence_data[i].confidence2,
                                'iou': cur_iou_data[i].iou,
                                'iou1': cur_iou_data[i].iou1,
                                'iou2': cur_iou_data[i].iou2,
                                'conf_iou': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou),
                                'conf_iou1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1),
                                'conf_iou2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2),
                                'conf_iou_pr': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision),
                                'conf_iou_pr1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1)*parseFloat(cur_precision_data[i].precision1),
                                'conf_iou_pr2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2)*parseFloat(cur_precision_data[i].precision2)})
        }
    }
    // plotting confidence*IOU*precision
    else if (cur_plot_quantity == 3) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'id': desired_classes[parseInt(cur_pred_id_data[i].id)],
                                'id1': desired_classes[parseInt(cur_pred_id_data[i].id1)],
                                'id2': desired_classes[parseInt(cur_pred_id_data[i].id2)],
                                'cq': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision),
                                'cq1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1)*parseFloat(cur_precision_data[i].precision),
                                'cq2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2)*parseFloat(cur_precision_data[i].precision),
                                'conf': cur_confidence_data[i].confidence,
                                'conf1': cur_confidence_data[i].confidence1,
                                'conf2': cur_confidence_data[i].confidence2,
                                'iou': cur_iou_data[i].iou,
                                'iou1': cur_iou_data[i].iou1,
                                'iou2': cur_iou_data[i].iou2,
                                'conf_iou': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou),
                                'conf_iou1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1),
                                'conf_iou2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2),
                                'conf_iou_pr': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision),
                                'conf_iou_pr1': parseFloat(cur_confidence_data[i].confidence1)*parseFloat(cur_iou_data[i].iou1)*parseFloat(cur_precision_data[i].precision1),
                                'conf_iou_pr2': parseFloat(cur_confidence_data[i].confidence2)*parseFloat(cur_iou_data[i].iou2)*parseFloat(cur_precision_data[i].precision2)})
        }
    }

    // Labels of row (y_translations) and columns (x_translations)
    var x_translations = d3.map(cur_cq_data, function(d){return d.x_tran;}).keys();
    var y_translations = d3.map(cur_cq_data, function(d){return d.y_tran;}).keys();
    x_translations.push(64);
    y_translations.push(64);

    // Build X scales and axis:
    var x_scale = d3.scaleBand()
                    .range([ 0, width ])
                    .domain(x_translations);
    var x_axis = d3.axisBottom(x_scale)
                    .tickValues(x_scale.domain().filter(function(d, i){ return !(i%8)}))
                    .tickSize(0);
    single_test_svg.append("g")
                    .style("font-size", 15)
                    .attr("transform", "translate(0," + height + ")")
                    .call(x_axis)
                    .select(".domain").remove()

    // Build Y scales and axis:
    var y_scale = d3.scaleBand()
                    .range([ 0, height ])
                    .domain(y_translations);
    var y_axis = d3.axisLeft(y_scale)
                    .tickValues(y_scale.domain().filter(function(d, i){ return !(i%8)}))
                    .tickSize(0);
    single_test_svg.append("g")
                    .style("font-size", 15)
                    .call(y_axis)
                    .select(".domain").remove()

    // click event
    var bb_info_exist = true;
    var mouseclick = function(d) {
        if (hover_turn_off == true) {
            hover_turn_off = false;
            d3.selectAll("#chosen_rect")
                .style("stroke", "none")
                .style("opacity", 1)
        }
        else {
            hover_turn_off = true;
            // make the selected rect to red
            d3.select(this)
                .style("stroke", "red")
                .style("opacity", 1)
                .attr('id', 'chosen_rect')
        }
    }
    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover = function(d) {
        if (hover_turn_off == false) {
            tooltip.style("opacity", 1)
            d3.select(this)
                .style("stroke", "black")
                .style("opacity", 1)
        }
    }
    var mousemove = function(d) {

        if (hover_turn_off == false) {

            // apply mask for the selected area
            tooltip_svg.selectAll("defs").remove();
            tooltip_svg.selectAll("rect").remove();
            show_extracted_area();

            var plot_variable;
            if (cur_plot_quantity == 0) {
                plot_variable = "Confidence";
            }
            else if (cur_plot_quantity == 1) {
                plot_variable = "Intersection over Union (IOU)";
            }
            else if (cur_plot_quantity == 2) {
                plot_variable = "Confidence*IOU";
            }
            else if (cur_plot_quantity == 3) {
                plot_variable = "Confidence*IOU*precision";
            }

            // append rect to the tooltip svg
            // tooltip_svg.selectAll("defs").remove();
            // tooltip_svg.selectAll("rect").remove();
            tooltip_svg.selectAll("text").remove();
            var row = parseInt(d.y_tran) + 64;
            var col = parseInt(d.x_tran) + 64;
            var object_class_index;
            if (cur_object_class == desired_classes[0]) {
                object_class_index = 0
            }
            else if (cur_object_class == desired_classes[1]) {
                object_class_index = 1
            }
            else if (cur_object_class == desired_classes[2]) {
                object_class_index = 2
            }
            else if (cur_object_class == desired_classes[3]) {
                object_class_index = 3
            }
            else if (cur_object_class == desired_classes[4]) {
                object_class_index = 4
            }
            sx = parseInt(cutout_pos_data[object_class_index][row*128+col].x_min);
            sy = parseInt(cutout_pos_data[object_class_index][row*128+col].y_min);

            // draw the ground truth bounding boxes
            var target_x = parseFloat(target_bb_data[object_class_index][row*128+col].x_min);
            var target_y = parseFloat(target_bb_data[object_class_index][row*128+col].y_min);
            var target_width = target_bb_data[object_class_index][row*128+col].width;
            var target_height = target_bb_data[object_class_index][row*128+col].height;
            var true_bb = [{x1: sx+target_x, y1: sy+target_y, width: target_width, height: target_height}];
            tooltip_svg.selectAll("rect1")
                        .data(true_bb)
                        .enter()
                        .append("rect")
                        .attr("x", function(d) { return d.x1 })
                        .attr("y", function(d) { return d.y1 })
                        .attr("width", function(d) { return d.width })
                        .attr("height", function(d) { return d.height })
                        .attr('fill', 'rgba(0,0,0,0)')
                        .attr('stroke', 'green')
                        .attr('stroke-width', 2)
                        .style("opacity", 1);

            // append label text
            var target_class_index = parseInt(target_id_data[object_class_index][row*128+col].id);
            tooltip_svg.append("text")
                        .attr("x", sx+target_x)
                        .attr("y", sy+target_y-2)
                        .text("True: "+desired_classes[target_class_index])
                        .attr('stroke', 'black')
                        .attr('stroke-width', 1)
                        .style("font-size", "8px");


            // draw the top three pred bounding boxes (drawn one by one for clicking events)
            // first pred bb
            var pred_x = parseFloat(cur_pred_bb_data[row*128+col].x_min);
            var pred_y = parseFloat(cur_pred_bb_data[row*128+col].y_min);
            var pred_width = parseFloat(cur_pred_bb_data[row*128+col].width);
            var pred_height = parseFloat(cur_pred_bb_data[row*128+col].height);
            var pred_bb = [{x1: sx+pred_x,
                            y1: sy+pred_y,
                            width: pred_width,
                            height: pred_height,
                            id: cur_cq_data[row*128+col].id,
                            bb_cq: cur_cq_data[row*128+col].cq,
                            bb_conf: cur_cq_data[row*128+col].conf,
                            bb_iou: cur_cq_data[row*128+col].iou,
                            bb_conf_iou: cur_cq_data[row*128+col].conf_iou,
                            bb_conf_iou_pr: cur_cq_data[row*128+col].conf_iou_pr,}];
            var pred_bb_clicked = false;
            var click_bb = function(d) {
                // when unselecting this bb
                if (pred_bb_clicked == true) {
                    pred_bb_clicked = false;
                    d3.select(this)
                        .style("stroke", function(d) { return colormap_single_test(d.bb_cq) })
                        .style("opacity", 1);
                    tooltip_text_svg.html("");
                }
                // when selecting this bb
                else {
                    pred_bb_clicked = true;
                    d3.select(this)
                        .style("stroke", "red")
                        .style("opacity", 1);
                    tooltip_text_svg
                        .html("Prediction 1</br>" +
                              "Object: " + d.id + "</br>" +
                              "Confidence: " + parseFloat(d.bb_conf).toFixed(2) + "</br>" +
                              "IOU: " + parseFloat(d.bb_iou).toFixed(2) + "</br>" +
                              "Conf*IOU: " + parseFloat(d.bb_conf_iou).toFixed(2) + "</br>" +
                              "Conf*IOU*Precision: " + parseFloat(d.bb_conf_iou_pr).toFixed(2)
                        )
                        .style("text-align", "left");
                }
            }
            tooltip_svg.selectAll("rect2")
                        .data(pred_bb)
                        .enter()
                        .append("rect")
                        .attr("x", function(d) { return d.x1 })
                        .attr("y", function(d) { return d.y1 })
                        .attr("width", function(d) { return d.width })
                        .attr("height", function(d) { return d.height })
                        .attr('fill', 'none')
                        .attr('stroke', function(d) { return colormap_single_test(d.bb_cq) })
                        .attr('stroke-width', 2)
                        .on("click", click_bb);

            // second pred bb
            var pred_x1 = parseFloat(cur_pred_bb_data[row*128+col].x_min1);
            var pred_y1 = parseFloat(cur_pred_bb_data[row*128+col].y_min1);
            var pred_width1 = parseFloat(cur_pred_bb_data[row*128+col].width1);
            var pred_height1 = parseFloat(cur_pred_bb_data[row*128+col].height1);
            var pred_bb1 = [{x1: sx+pred_x1,
                             y1: sy+pred_y1,
                             width: pred_width1,
                             height: pred_height1,
                             id: cur_cq_data[row*128+col].id1,
                             bb_cq: cur_cq_data[row*128+col].cq1,
                             bb_conf: cur_cq_data[row*128+col].conf1,
                             bb_iou: cur_cq_data[row*128+col].iou1,
                             bb_conf_iou: cur_cq_data[row*128+col].conf_iou1,
                             bb_conf_iou_pr: cur_cq_data[row*128+col].conf_iou_pr1}];
            var pred_bb_clicked1 = false;
            var click_bb1 = function(d) {
                // when unselecting this bb
                if (pred_bb_clicked1 == true) {
                    pred_bb_clicked1 = false;
                    d3.select(this)
                        .style("stroke", function(d) { return colormap_single_test(d.bb_cq) })
                        .style("opacity", 1);
                    tooltip_text_svg.html("");
                }
                // when selecting this bb
                else {
                    pred_bb_clicked1 = true;
                    d3.select(this)
                        .style("stroke", "red")
                        .style("opacity", 1);
                    tooltip_text_svg
                        .html("Prediction 2</br>" +
                              "Object: " + d.id + "</br>" +
                              "Confidence: " + parseFloat(d.bb_conf).toFixed(2) + "</br>" +
                              "IOU: " + parseFloat(d.bb_iou).toFixed(2) + "</br>" +
                              "Conf*IOU: " + parseFloat(d.bb_conf_iou).toFixed(2) + "</br>" +
                              "Conf*IOU*Precision: " + parseFloat(d.bb_conf_iou_pr).toFixed(2)
                        )
                        .style("text-align", "left");
                }
            }
            tooltip_svg.selectAll("rect2")
                        .data(pred_bb1)
                        .enter()
                        .append("rect")
                        .attr("x", function(d) { return d.x1 })
                        .attr("y", function(d) { return d.y1 })
                        .attr("width", function(d) { return d.width })
                        .attr("height", function(d) { return d.height })
                        .attr('fill', 'none')
                        .attr('stroke', function(d) { return colormap_single_test(d.bb_cq) })
                        .attr('stroke-width', 2)
                        .on("click", click_bb1);

            var pred_x2 = parseFloat(cur_pred_bb_data[row*128+col].x_min2);
            var pred_y2 = parseFloat(cur_pred_bb_data[row*128+col].y_min2);
            var pred_width2 = parseFloat(cur_pred_bb_data[row*128+col].width2);
            var pred_height2 = parseFloat(cur_pred_bb_data[row*128+col].height2);
            var pred_bb2 = [{x1: sx+pred_x2,
                             y1: sy+pred_y2,
                             width: pred_width2,
                             height: pred_height2,
                             id: cur_cq_data[row*128+col].id2,
                             bb_cq: cur_cq_data[row*128+col].cq2,
                             bb_conf: cur_cq_data[row*128+col].conf2,
                             bb_iou: cur_cq_data[row*128+col].iou2,
                             bb_conf_iou: cur_cq_data[row*128+col].conf_iou2,
                             bb_conf_iou_pr: cur_cq_data[row*128+col].conf_iou_pr2}];
            var pred_bb_clicked2 = false;
            var click_bb2 = function(d) {
                // when unselecting this bb
                if (pred_bb_clicked2 == true) {
                    pred_bb_clicked2 = false;
                    d3.select(this)
                        .style("stroke", function(d) { return colormap_single_test(d.bb_cq) })
                        .style("opacity", 1);
                    tooltip_text_svg.html("");
                }
                // when selecting this bb
                else {
                    pred_bb_clicked2 = true;
                    d3.select(this)
                        .style("stroke", "red")
                        .style("opacity", 1);
                    tooltip_text_svg
                        .html("Prediction 3</br>" +
                              "Object: " + d.id + "</br>" +
                              "Confidence: " + parseFloat(d.bb_conf).toFixed(2) + "</br>" +
                              "IOU: " + parseFloat(d.bb_iou).toFixed(2) + "</br>" +
                              "Conf*IOU: " + parseFloat(d.bb_conf_iou).toFixed(2) + "</br>" +
                              "Conf*IOU*Precision: " + parseFloat(d.bb_conf_iou_pr).toFixed(2)
                        )
                        .style("text-align", "left");
                }
            }
            tooltip_svg.selectAll("rect2")
                        .data(pred_bb2)
                        .enter()
                        .append("rect")
                        .attr("x", function(d) { return d.x1 })
                        .attr("y", function(d) { return d.y1 })
                        .attr("width", function(d) { return d.width })
                        .attr("height", function(d) { return d.height })
                        .attr('fill', 'none')
                        .attr('stroke', function(d) { return colormap_single_test(d.bb_cq) })
                        .attr('stroke-width', 2)
                        .on("click", click_bb2);

            // append label text for all three bounding boxes
            tooltip_svg.append("text")
                        .attr("x", sx+pred_x+2)
                        .attr("y", sy+pred_y+pred_height-2)
                        .text("1")
                        .attr('stroke', "black")
                        .attr('stroke-width', 1)
                        .style("font-size", "8px");
            tooltip_svg.append("text")
                        .attr("x", sx+pred_x1+2)
                        .attr("y", sy+pred_y1+pred_height1-2)
                        .text("2")
                        .attr('stroke', "black")
                        .attr('stroke-width', 1)
                        .style("font-size", "8px");
            tooltip_svg.append("text")
                        .attr("x", sx+pred_x2+2)
                        .attr("y", sy+pred_y2+pred_height2-2)
                        .text("3")
                        .attr('stroke', "black")
                        .attr('stroke-width', 1)
                        .style("font-size", "8px");

            // show the values based on selection and no selection
            // if (pred_bb_clicked == false && pred_bb_clicked1 == false && pred_bb_clicked2 == false) {
            //     tooltip_text_svg.selectAll("text").remove();
            //     text1 = "<p style='font-size:11px'>"
            //             + "Box 1: " + desired_classes[cur_pred_id_data[row*128+col].id] + ", " + plot_variable + "(" + d.x_tran + ', ' + d.y_tran + ') = ' + parseFloat(d.cq).toFixed(3) + '</br>'
            //             + "Box 2: " + desired_classes[cur_pred_id_data[row*128+col].id1] + ", " +  plot_variable + "(" + d.x_tran + ', ' + d.y_tran + ') = ' + parseFloat(d.cq1).toFixed(3) + '</br>'
            //             + "Box 3: " + desired_classes[cur_pred_id_data[row*128+col].id2] + ", " +  plot_variable + "(" + d.x_tran + ', ' + d.y_tran + ') = ' + parseFloat(d.cq2).toFixed(3) + '</br></p>';
            //     tooltip_text_svg.html(text1);
            // }
            // else {
            //     tooltip_text_svg.selectAll("text").remove();
            // }


            // apply mask for the selected area
            // show_extracted_area();
        }
    }
    var mouseleave = function(d) {
        if (hover_turn_off == false) {
            tooltip.style("opacity", 0)
            d3.select(this)
                .style("stroke", "none")
                .style("opacity", 1)
            if (bb_info_exist == true) {
                tooltip_text_svg.html("");
            }
        }
    }

    // add the squares
    // heat map is created differently by input
    single_test_svg.selectAll('rect').remove();
    single_test_svg.selectAll('rect')
                .data(cur_cq_data, function(d) {return d.cq;})
                .enter()
                .append("rect")
                .attr("x", function(d) { return x_scale(d.x_tran) })
                .attr("y", function(d) { return y_scale(d.y_tran) })
                .attr("rx", 0)
                .attr("ry", 0)
                .attr("width", x_scale.bandwidth() )
                .attr("height", y_scale.bandwidth() )
                .style("fill", function(d) { return colormap_single_test(d.cq)} )
                .style("stroke-width", 2)
                .style("stroke", "none")
                .style("opacity", 1)
                .on("click", mouseclick)
                .on("mouseover", mouseover)
                .on("mousemove", mousemove)
                .on("mouseleave", mouseleave)

}

// helper function for plotting single test heatmap
function show_extracted_area () {
    // set up defs
    var defs = tooltip_svg.append('defs')
    // define a clip path area
    defs.append('clipPath')
        .attr('id', 'rect_clip')
        .call(append_rect)

    // add the image, which is clipped via the clip-path attribute
    tooltip_svg.append('image')
                .attr('xlink:href', image_path)
                .attr('x', 0)
                .attr('y', 0)
                // clip the image using id
                .attr('clip-path', 'url(#rect_clip)')
                .attr('opacity', 1);
}

// helper function for plotting single test heatmap
// create a clip path area
function append_rect (selection) {
    selection.append('rect')
                .attr('x', sx)
                .attr('y', sy)
                .attr('width', 128)
                .attr('height', 128)
  }

function plot_heatmap_canvas(image_index,
                            jittering,
                            sort_method,
                            plot_quantity,
                            target_id_data,
                            target_bb_data,
                            pred_id_data_sorted_by_conf,
                            pred_bb_data_sorted_by_conf,
                            confidence_data_sorted_by_conf,
                            iou_data_sorted_by_conf,
                            pred_id_data_sorted_by_iou,
                            pred_bb_data_sorted_by_iou,
                            confidence_data_sorted_by_iou,
                            iou_data_sorted_by_iou,
                            precision_data,
                            data_container,
                            tooltip,
                            cutout_pos_data) {

    // choose the set of data by input
    var cur_pred_id_data;
    var cur_pred_bb_data;
    var cur_confidence_data;
    var cur_iou_data;
    var cur_precision_data = precision_data[image_index][jittering];
    // sorted by conf
    if (sort_method == 0) {
        cur_pred_id_data = pred_id_data_sorted_by_conf[image_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_conf[image_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_conf[image_index][jittering]
        cur_iou_data = iou_data_sorted_by_conf[image_index][jittering]
    // sorted by iou
    } else if (sort_method == 1) {
        cur_pred_id_data = pred_id_data_sorted_by_iou[image_index][jittering]
        cur_pred_bb_data = pred_bb_data_sorted_by_iou[image_index][jittering]
        cur_confidence_data = confidence_data_sorted_by_iou[image_index][jittering]
        cur_iou_data = iou_data_sorted_by_iou[image_index][jittering]
    }

    // determine the quantity that we are plotting
    var cur_cq_data = [];
    // plotting confidence
    if (plot_quantity == 0) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'cq': cur_confidence_data[i].confidence})
        }
    }
    // plotting iou
    else if (plot_quantity == 1) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'cq': cur_iou_data[i].iou})
        }
    }
    // plotting confidence*IOU
    else if (plot_quantity == 2) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'cq': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)})
        }
    }
    // plotting confidence*IOU*precision
    else if (plot_quantity == 3) {
        for (var i = 0; i < cur_confidence_data.length; i++) {
            cur_cq_data.push({'x_tran': cur_confidence_data[i].x_tran,
                                'y_tran': cur_confidence_data[i].y_tran,
                                'cq': parseFloat(cur_confidence_data[i].confidence)*parseFloat(cur_iou_data[i].iou)*parseFloat(cur_precision_data[i].precision)})
        }
    }

    // Build X scales and axis:
    var x_translations = d3.map(cur_cq_data, function(d){return d.x_tran;}).keys()
    var y_translations = d3.map(cur_cq_data, function(d){return d.y_tran;}).keys()
    x_translations.push(64);
    y_translations.push(64);

    var x_scale = d3.scaleBand()
                    .domain(x_translations)
                    .range([ 0, width ]);

    var x_axis = d3.axisBottom(x_scale)
                    .tickValues(x_scale.domain().filter(function(d, i){ return !(i%8)}))
                    .tickSize(0);

    // Build Y scales and axis:
    var y_scale = d3.scaleBand()
                    .domain(y_translations)
                    .range([ 0, height ]);

    var y_axis = d3.axisLeft(y_scale)
                    .tickValues(y_scale.domain().filter(function(d, i){ return !(i%8)}))
                    .tickSize(0);

    // bind the data to our dom-attached container
    var data_binding = data_container.selectAll("custom.rect")
                                     .data(cur_cq_data, function(d) { return d.cq; });

    // update existing element to have size 15 and fill green
    // data_binding.attr("size", 15)
    //             .attr("fillStyle", "green");
    // for exiting elements, remove them
    data_binding.exit().remove();

    // for new elements, create a 'custom' dom node, of class rect with the appropriate rect attributes
    var new_data_binding = data_binding.enter()
                                        .append("custom")
                                        .classed("rect", true)
                                        .attr("x", x_scale)
                                        .attr("y", y_scale)
                                        .attr("size", rect_size)
                                        .attr("fillStyle", function(d) { return colormap_single_test(d.cq)} );


    // console.log(cur_cq_data[0]);
    new_data_binding.merge(data_binding)
                // .transition(t)
                .attr('fillStyle', function(d) { return colormap_single_test(d.cq) });

    draw_canvas(cur_cq_data);
}


function draw_canvas(cur_cq_data) {

    // canvas color (white/clear)
    context.fillStyle = "#ff0";
    context.rect(0, 0, my_canvas.attr("width"), my_canvas.attr("height"));
    context.fill();

    var elements = data_container.selectAll("custom.rect");

    var i = 0;
    elements.each(function(d) {
        var node = d3.select(this);
        node.attr("x", (parseInt(cur_cq_data[i].x_tran)+64)*rect_size);
        node.attr("y", (parseInt(cur_cq_data[i].y_tran)+64)*rect_size);
        i = i + 1;
    })

    elements.each(function(d) {
      var node = d3.select(this);

      context.beginPath();
      context.fillStyle = node.attr("fillStyle");
      context.rect(node.attr("x"), node.attr("y"), node.attr("size"), node.attr("size"));
      context.fill();
      context.closePath();
    });
}
