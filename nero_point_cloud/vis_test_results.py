import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pol2cart(length, angle):
    x = int(length * math.cos(math.radians(angle)))
    y = int(length * math.sin(math.radians(angle)))
    return (x, y)

# only work on square images (single img_size)
def cart2img(cart, img_size):
    x_img = cart[0] + img_size // 2
    y_img = cart[1] + img_size // 2

    return (x_img, y_img)



# create raster image that holds custom displacement of results
def get_raster_image(result, plane, type, size=(400, 400)):
    # get all the useful information
    all_planes = result['all_planes']
    plane_idx = np.where(all_planes==plane)[0][0]
    all_axis_angles = result['all_axis']
    all_rot_angles = result['all_angles']

    if type == 'instance':
        accuracies = result['instance_accuracies'][plane_idx]
    elif type == 'class':
        accuracies = result['class_accuracies'][plane_idx]

    assert (
        len(all_axis_angles) == accuracies.shape[0]
        and len(all_rot_angles) == accuracies.shape[1]
    )

    image_np = np.zeros((len(all_rot_angles)*2, len(all_rot_angles)*2))

    for i, cur_axis_angle in enumerate(all_axis_angles):
        for j, cur_rot_angle in enumerate(all_rot_angles):
            # coordinate in raster image
            cartesian_coordinate = pol2cart(j, cur_axis_angle)
            # cartesian coordinate to image coordinate
            image_coordinate = cart2img(cartesian_coordinate, len(image_np))
            image_np[image_coordinate] = accuracies[i, j]

    # resize and convert to image
    image_np = np.kron(image_np, np.ones((50, 50)))
    image_pil = Image.fromarray(image_np*255).convert("L")

    return image_pil


result_dir = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_point_cloud/output/'
result_names = [
    'point_transformer_model_rot_False_e_20.npz',
    'point_transformer_model_rot_True_e_100.npz',
]
img_dir = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_point_cloud/output/'

for i, cur_name in enumerate(result_names):
    result = np.load(os.path.join(result_dir, cur_name))
    raster_img = get_raster_image(result, 'xy', 'instance')
    img_path = os.path.join(img_dir, f'{cur_name}.jpg')
    raster_img.save(img_path)


# plot_types = ['heatmap', 'polar_heatmap', 'voronoi']
# plot_types = ['heatmap']

# fig, ax = plt.subplots(nrows=len(plot_types), ncols=len(result_names)*3)

# for k, plot_type in enumerate(plot_types):
#     for i, cur_name in enumerate(result_names):
#         # load the result
#         result = np.load(os.path.join(result_dir, cur_name))
#         all_planes = result['all_planes']
#         all_axis = result['all_axis']
#         all_angles = result['all_angles']
#         instance_accuracies = result['instance_accuracies']
#         class_accuracies = result['class_accuracies']

#         for j, cur_plane in enumerate(all_planes):
#             cur_instance_accuracy = instance_accuracies[j]
#             cur_class_accuracy = class_accuracies[j]

#             if plot_type == 'heatmap':
#                 ax[i*3+j].imshow(cur_instance_accuracy)
#             else:
#                 continue
#             # elif plot_type == 'polar_heatmap':
#             #     ax[k, i*3+j].imshow(cur_instance_accuracy)

# plt.imshow(cur_instance_accuracy)
# plt.show()