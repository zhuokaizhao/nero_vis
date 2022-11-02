// append the svg object to the body of the page
var normal_single_svg = d3.select("#normal_heatmap")
                            .append("svg")
                            .attr("width", width + margin.left + margin.right)
                            .attr("height", height + margin.top + margin.bottom)
                            .append("g")
                            .attr("transform",
                                    "translate(" + margin.left + "," + margin.top + ")");

// Add title to graph
normal_single_svg.append("text")
                    .attr("x", title_x)
                    .attr("y", title_y)
                    .attr("text-anchor", "center")
                    .style("font-size", "22px")
                    .text("Non-shift-equavariant model");


// Add subtitle to graph
if (cur_sort_method == 0) {
    if (cur_plot_quantity == 0) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence score of the most confident prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 1) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting IOU of the most confident prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 2) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU of the most confident prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 3) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU*precision of the most confident prediction").attr('id', 'normal_subtitle');
    }
} else if (cur_sort_method == 1) {
    if (cur_plot_quantity == 0) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence score of the most correct prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 1) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting IOU of the most correct prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 2) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU of the most correct prediction").attr('id', 'normal_subtitle');
    } else if (cur_plot_quantity == 3) {
        normal_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU*precision of the most correct prediction").attr('id', 'normal_subtitle');
    }
}


