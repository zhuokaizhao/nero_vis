// Global variales for both normal and shift-equivariant model js scripts

// set the dimensions and margins of the graph
var margin = {top: 100, right: 25, bottom: 25, left: 25};
var width = 300
var height = 300

var result_jittering_levels = ['0', '20', '40', '60', '80', '100'];
var desired_classes = ['car', 'bottle', 'cup', 'chair', 'book'];
var coco_classes = ['person',
                    'bicycle',
                    'car',
                    'motorcycle',
                    'airplane',
                    'bus',
                    'train',
                    'truck',
                    'boat',
                    'traffic light',
                    'fire hydrant',
                    'stop sign',
                    'parking meter',
                    'bench',
                    'bird',
                    'cat',
                    'dog',
                    'horse',
                    'sheep',
                    'cow',
                    'elephant',
                    'bear',
                    'zebra',
                    'giraffe',
                    'backpack',
                    'umbrella',
                    'handbag',
                    'tie',
                    'suitcase',
                    'frisbee',
                    'skis',
                    'snowboard',
                    'sports ball',
                    'kite',
                    'baseball bat',
                    'baseball glove',
                    'skateboard',
                    'surfboard',
                    'tennis racket',
                    'bottle',
                    'wine glass',
                    'cup',
                    'fork',
                    'knife',
                    'spoon',
                    'bowl',
                    'banana',
                    'apple',
                    'sandwich',
                    'orange',
                    'broccoli',
                    'carrot',
                    'hot dog',
                    'pizza',
                    'donut',
                    'cake',
                    'chair',
                    'couch',
                    'potted plant',
                    'bed',
                    'dining table',
                    'toilet',
                    'tv',
                    'laptop',
                    'mouse',
                    'remote',
                    'keyboard',
                    'cell phone',
                    'microwave',
                    'oven',
                    'toaster',
                    'sink',
                    'refrigerator',
                    'book',
                    'clock',
                    'vase',
                    'scissors',
                    'teddy bear',
                    'hair drier',
                    'toothbrush']
var data_dir = "./data/"
var test_plot_quantities = ['precision', 'recall', 'AP', 'f1'];
var single_test_plot_quantities = ['confidence', 'IOU', 'Confidence*IOU', 'Confidence*IOU*precision'];

// heatmap plot title and subtitle offsets
var title_x = 25;
var title_y = -40;
var title_x1 = -20;
var title_y1 = -40;
var subtitle_x = -20;
var subtitle_y = -20;

// initialize plotting settings
var cur_vis_mode = 'test';
var cur_si_method = 'si';
var cur_object_class = 'car';
var cur_plot_quantity = 'AP';
var cur_jittering_level = '0';

document.getElementById('shift_equivariant_method').style.display = 'none';

// data associated with tests
var normal_test_data = {};
var si_test_data = {};
var pt_test_data = {};

// data associated with single-tests
// global loaded data variables
var target_id_data = [];
var target_bb_data = [];
var cutout_pos_data = [];

// normal model results data loading
var normal_pred_id_data_sorted_by_conf = [];
var normal_pred_bb_data_sorted_by_conf = [];
var normal_confidence_data_sorted_by_conf = [];
var normal_iou_data_sorted_by_conf = [];
var normal_pred_id_data_sorted_by_iou = [];
var normal_pred_bb_data_sorted_by_iou = [];
var normal_confidence_data_sorted_by_iou = [];
var normal_iou_data_sorted_by_iou = [];
var normal_pred_id_data_sorted_by_conf_iou = [];
var normal_pred_bb_data_sorted_by_conf_iou = [];
var normal_confidence_data_sorted_by_conf_iou = [];
var normal_iou_data_sorted_by_conf_iou = [];
var normal_pred_id_data_sorted_by_conf_iou_pr = [];
var normal_pred_bb_data_sorted_by_conf_iou_pr = [];
var normal_confidence_data_sorted_by_conf_iou_pr = [];
var normal_iou_data_sorted_by_conf_iou_pr = [];
var normal_precision_data = [];

// shift-invariant model results data loading
var si_pred_id_data_sorted_by_conf = [];
var si_pred_bb_data_sorted_by_conf = [];
var si_confidence_data_sorted_by_conf = [];
var si_iou_data_sorted_by_conf = [];
var si_pred_id_data_sorted_by_iou = [];
var si_pred_bb_data_sorted_by_iou = [];
var si_confidence_data_sorted_by_iou = [];
var si_iou_data_sorted_by_iou = [];
var si_pred_id_data_sorted_by_conf_iou = [];
var si_pred_bb_data_sorted_by_conf_iou = [];
var si_confidence_data_sorted_by_conf_iou = [];
var si_iou_data_sorted_by_conf_iou = [];
var si_pred_id_data_sorted_by_conf_iou_pr = [];
var si_pred_bb_data_sorted_by_conf_iou_pr = [];
var si_confidence_data_sorted_by_conf_iou_pr = [];
var si_iou_data_sorted_by_conf_iou_pr = [];
var si_precision_data = [];

// pre-trained model results data loading
var pt_pred_id_data_sorted_by_conf = [];
var pt_pred_bb_data_sorted_by_conf = [];
var pt_confidence_data_sorted_by_conf = [];
var pt_iou_data_sorted_by_conf = [];
var pt_pred_id_data_sorted_by_iou = [];
var pt_pred_bb_data_sorted_by_iou = [];
var pt_confidence_data_sorted_by_iou = [];
var pt_iou_data_sorted_by_iou = [];
var pt_pred_id_data_sorted_by_conf_iou = [];
var pt_pred_bb_data_sorted_by_conf_iou = [];
var pt_confidence_data_sorted_by_conf_iou = [];
var pt_iou_data_sorted_by_conf_iou = [];
var pt_pred_id_data_sorted_by_conf_iou_pr = [];
var pt_pred_bb_data_sorted_by_conf_iou_pr = [];
var pt_confidence_data_sorted_by_conf_iou_pr = [];
var pt_iou_data_sorted_by_conf_iou_pr = [];
var pt_precision_data = [];

// initiate a tooltip
var hover_turn_off = false;
var tooltip = d3.select("#tooltip_div")
                .append("div")
                .style("opacity", 0)
                .attr("class", "tooltip");

var tooltip_text_svg = d3.select("#tooltip_text_div");

// always show the original image as the background of tooltip
var image_path;
var tooltip_svg = d3.select("#tooltip_div")
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                // .style("border", "1px solid black")
                .style("opacity", 1.0);

var sx;
var sy;

// append the svg object to the body of the page
var normal_svg = d3.select("#normal_heatmap")
                    .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform",
                            "translate(" + margin.left + "," + margin.top + ")");

var si_svg = d3.select("#si_heatmap")
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");

var pt_svg = d3.select("#pt_heatmap")
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .append("g")
                .attr("transform",
                        "translate(" + margin.left + "," + margin.top + ")");

// load the dataset
if (cur_vis_mode == 'test') {
    // make selection of the set of plot quantities
    document.getElementById('single_test_plot_quantities').style.display = 'none';
    // load the dataset
    load_test_data(plot=true);
    load_single_test_data()
}

// block that uses canvas heatmap
// // div base that we are putting canvas on
// var base = d3.select("#heatmap");
// // instead of appending div, append canvas
// // size of each rect in the heatmap
// var rect_size = 2;
// // number of rect in a heatmap's row/col
// var num_rect = 127;
// var my_canvas = base.append("canvas")
//                     .attr("id", "my_canvas")
//                     .attr("width", num_rect*rect_size)
//                     .attr("height", num_rect*rect_size);

// var context = my_canvas.node().getContext("2d");

// // custom element that will not be attached to the DOM
// var detached_container = document.createElement("custom");
// // d3 selection for the detached container, that will be attached to the DOM
// var data_container = d3.select(detached_container);
