// append the svg object to the body of the page
var si_single_svg = d3.select("#si_heatmap")
                        .append("svg")
                        .attr("width", width + margin.left + margin.right)
                        .attr("height", height + margin.top + margin.bottom)
                        .append("g")
                        .attr("transform",
                                "translate(" + margin.left + "," + margin.top + ")");

// Add title to graph
si_single_svg.append("text")
            .attr("x", title_x1)
            .attr("y", title_y1)
            .attr("text-anchor", "center")
            .style("font-size", "22px")
            .text("Shift-equavariant model (" + cur_si_method + ")").attr('id', 'si_title');


// Add subtitle to graph based on plot quantity
if (cur_sort_method == 0) {
    if (cur_plot_quantity == 0) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence score of the most confident prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 1) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting IOU of the most confident prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 2) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU of the most confident prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 3) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU*precision of the most confident prediction").attr('id', 'si_subtitle');
    }
} else if (cur_sort_method == 1) {
    if (cur_plot_quantity == 0) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence score of the most correct prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 1) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting IOU of the most correct prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 2) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU of the most correct prediction").attr('id', 'si_subtitle');
    } else if (cur_plot_quantity == 3) {
        si_single_svg.append("text")
                        .attr("x", subtitle_x)
                        .attr("y", subtitle_y)
                        .attr("text-anchor", "center")
                        .style("font-size", "14px")
                        .style("fill", "grey")
                        .style("max-width", 400)
                        .text("Plotting confidence*IOU*precision of the most correct prediction").attr('id', 'si_subtitle');
    }
}
