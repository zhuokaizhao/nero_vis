import os
import numpy as np
import matplotlib.pyplot as plt


result_dir = '/home/zhuokai/Desktop/UChicago/Research/nero_vis/nero_point_cloud/output/'
result_names = [
    'point_transformer_model_rot_False_e_20.npz',
    'point_transformer_model_rot_True_e_20.npz',
]
# plot_types = ['heatmap', 'polar_heatmap', 'voronoi']
plot_types = ['heatmap']

fig, ax = plt.subplots(nrows=len(plot_types), ncols=len(result_names)*3)

for k, plot_type in enumerate(plot_types):
    for i, cur_name in enumerate(result_names):
        # load the result
        result = np.load(os.path.join(result_dir, cur_name))
        all_planes = result['all_planes']
        all_axis = result['all_axis']
        all_angles = result['all_angles']
        instance_accuracies = result['instance_accuracies']
        class_accuracies = result['class_accuracies']

        for j, cur_plane in enumerate(all_planes):
            cur_instance_accuracy = instance_accuracies[j]
            cur_class_accuracy = class_accuracies[j]

            if plot_type == 'heatmap':
                ax[i*3+j].imshow(cur_instance_accuracy)
            else:
                continue
            # elif plot_type == 'polar_heatmap':
            #     ax[k, i*3+j].imshow(cur_instance_accuracy)

plt.imshow(cur_instance_accuracy)
plt.show()