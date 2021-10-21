var base = d3.select("#vis");

var chart = base.append("canvas")
                .attr("width", 400)
                .attr("height", 400);
var context = chart.node().getContext("2d");

var dataContainer = base.append("custom");

var colormap = d3.scaleSequential()
                .domain([0,1])
                .interpolator(d3.interpolateViridis);


// dummy dataset with the same format as real ones
var my_data1 = [{x: 100, y: 100, cq: 0.0}, {x: 200, y: 200, cq:0.5}, {x: 300, y: 300, cq:1.0}];
var my_data2 = [{x: 0, y: 0, cq: 0.0}, {x: 150, y: 150, cq:0.5}, {x: 250, y: 250, cq:1.0}];

drawCustom(my_data1);
drawCustom(my_data2);

function drawCustom(data) {
    var scale = d3.scaleLinear()
                    .range([0, 390])
                    .domain(d3.extent(data));

    var dataBinding = dataContainer.selectAll("custom.rect")
                                    .data(data, function(d) { return d.cq; });

    // dataBinding.attr("size", 15)
    //             .attr("fillStyle", "green");

    dataBinding.enter()
                .append("custom")
                .classed("rect", true)
                .attr("x", scale)
                .attr("y", scale)
                .attr("size", 10)
                .attr("fillStyle", function(d) { return colormap(d.cq)} );

    // for exiting elements, change the size to 5 and make them grey.
    // dataBinding.exit().remove();

    drawCanvas(data);
}

function drawCanvas(data) {

    // yellow canvas for debugging
    context.fillStyle = "#ff0";
    context.rect(0, 0, chart.attr("width"), chart.attr("height"));
    context.fill();

    var elements = dataContainer.selectAll("custom.rect");

    // I feel that this is done so stupidly
    var i = 0;
    elements.each(function(d) {
        var node = d3.select(this);
        node.attr("x", data[i].x);
        node.attr("y", data[i].y);
        i = i + 1;
    })

    elements.each(function(d) {
        var node = d3.select(this);
        context.beginPath();
        context.fillStyle = node.attr("fillStyle");
        context.rect(node.attr("x"), node.attr("y"), node.attr("size"), node.attr("size"));
        context.fill();
        context.closePath();
    })
}
