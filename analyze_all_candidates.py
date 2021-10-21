# The script analyzes the all-candidate test results
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import manifold
from sklearn.manifold import TSNE
import umap
from skimage.metrics import structural_similarity as compute_ssim

def compute_mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar" the two images are
	return err


jittering = '33'
desired_classes = ['car', 'bottle', 'cup', 'chair', 'book']
evaluated_quantities = ['all_precision', 'all_recall', 'all_F_measure']
# Dimension reduction algorithms
dimension_reduction_algos = ['PCA', 'ICA', 'ISOMAP', 't-SNE', 'UMAP']
dimension_reduction_algos = ['PCA', 'ICA']

for quantity in evaluated_quantities:

    # load the test result
    test_result_path = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/faster_rcnn/output/test/normal/faster_rcnn_resnet50_test_{jittering}-jittered_normal.npz'

    if quantity == 'all_precision':
        test_quantity = 'all_mean_precision'
    elif quantity == 'all_recall':
        test_quantity = 'all_mean_recall'
    elif quantity == 'all_F_measure':
        test_quantity = 'all_mean_f1'

    test_result = np.load(test_result_path)[test_quantity]

    # load the single-test result
    # initialize single test result dictionaries
    all_single_test_result = {}
    all_mse = {}
    all_ssim = {}

    for cur_class in desired_classes:
        # load the single test result
        single_test_dir = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/faster_rcnn/output/single_test_all_candidates/normal/{cur_class}'
        all_candidates_path = f'/home/zhuokai/Desktop/nvme1n1p1/Data/processed_coco/small_dataset/shift_equivariance/single_test/all_candidates/{cur_class}_single_test_candidates_val.npz'

        all_image_indices = np.load(all_candidates_path)['image_index']
        all_label_indices = np.load(all_candidates_path)['key_label_index']

        if len(all_image_indices) != len(all_label_indices):
            raise Exception(f'Unmatched number of image indices and label indices')

        single_test_result = []
        mse = []
        ssim = []

        for j in range(len(all_image_indices)):
            # for each candidate image in this class
            image_index = all_image_indices[j]
            label_index = all_label_indices[j]

            single_test_path = os.path.join(single_test_dir, f'faster_rcnn_resnet50_single-test-all-candidates_{jittering}-jittered_normal_{image_index}_{label_index}.npz')

            cur_single_result = np.load(single_test_path)[quantity]
            single_test_result.append(cur_single_result.flatten())

            # compute the image difference between each single test heatmap and the aggregated heatmap
            test_index = desired_classes.index(cur_class)
            cur_test_result = test_result[:, :, test_index]
            # mse
            cur_mse = compute_mse(cur_single_result, cur_test_result)
            mse.append(cur_mse)
            # ssim
            cur_ssim = compute_ssim(cur_single_result, cur_test_result, data_range=cur_single_result.max() - cur_single_result.min())
            ssim.append(cur_ssim)

        all_single_test_result[cur_class] = np.array(single_test_result)
        all_mse[cur_class] = np.array(mse)
        all_ssim[cur_class] = np.array(ssim)

    # print(all_single_test_result[0]['car'][:5])
    # print(all_single_test_result[0]['bottle'][:5])
    # exit()
    # make the plot
    for similarity in ['mse', 'ssim']:
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        fig, axes = plt.subplots(nrows=len(desired_classes), ncols=len(dimension_reduction_algos), figsize=(3*len(dimension_reduction_algos)+3, 3*len(desired_classes)-2))

        for i, dimension_reduction in enumerate(dimension_reduction_algos):

            print(f'\nDimension reduction algo: {dimension_reduction}')

            for j, cur_class in enumerate(desired_classes):

                # run dimension reduction on all single test result
                # dimension reduced from 841 to 2
                if dimension_reduction == 'PCA':
                    # pca = PCA(n_components=2, svd_solver='full')
                    # low_dimension = pca.fit_transform(all_single_test_result[cur_class])
                    # print(pca.explained_variance_[:2])
                    # print(pca.components_[:2])
                    # print(low_dimension[0])

                    # get covariance matrix
                    cov_matrix = np.cov(all_single_test_result[cur_class].T)
                    # eigendecomposition
                    values, vectors = np.linalg.eig(cov_matrix)
                    values = np.real(values)
                    vectors = np.real(vectors)

                    # project onto principle components
                    low_dimension = np.zeros((len(all_single_test_result[cur_class]), 2))
                    low_dimension[:, 0] = all_single_test_result[cur_class].dot(vectors.T[0])
                    low_dimension[:, 1] = all_single_test_result[cur_class].dot(vectors.T[1])
                    print(values[:2])
                    # print(vectors[:2])
                    # print(low_dimension[0])
                    # exit()

                    # explained_variances = []
                    # for i in range(len(values)):
                    #     explained_variances.append(values[i] / np.sum(values))

                    # print(np.sum(explained_variances), '\n', explained_variances)

                elif dimension_reduction == 'ICA':
                    ica = FastICA(n_components=2, random_state=12)
                    low_dimension = ica.fit_transform(all_single_test_result[cur_class])
                elif dimension_reduction == 'ISOMAP':
                    low_dimension = manifold.Isomap(n_neighbors=5, n_components=2, n_jobs=-1).fit_transform(all_single_test_result[cur_class])
                elif dimension_reduction == 't-SNE':
                    low_dimension = TSNE(n_components=2, n_iter=300).fit_transform(all_single_test_result[cur_class])
                elif dimension_reduction == 'UMAP':
                    low_dimension = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(all_single_test_result[cur_class])

                # for k in range(len(low_dimension))
                print(f'{dimension_reduction} for class {cur_class} completed')

                if similarity == 'mse':
                    sim_score = all_mse[cur_class]
                elif similarity == 'ssim':
                    sim_score = all_ssim[cur_class]

                if len(dimension_reduction_algos) == 1 and len(desired_classes) == 1:
                    axes.scatter(low_dimension[:, 0], low_dimension[:, 1],
                                c=sim_score, edgecolor='none', alpha=1, label=cur_class, cmap='viridis', marker='o')
                    axes.axis('equal')
                    # axes.set_xlabel('component 1')
                    # axes.set_ylabel('component 2')
                elif len(dimension_reduction_algos) == 1 and len(desired_classes) != 1:
                    axes[j].scatter(low_dimension[:, 0], low_dimension[:, 1],
                                c=sim_score, edgecolor='none', alpha=1, label=cur_class, cmap='viridis', marker='o')
                    axes[j].axis('equal')
                    axes[j].set_xlabel('component 1')
                    axes[j].set_ylabel('component 2')

                elif len(desired_classes) == 1 and len(dimension_reduction_algos) != 1:
                    axes[i].scatter(low_dimension[:, 0], low_dimension[:, 1],
                                c=sim_score, edgecolor='none', alpha=1, label=cur_class, cmap='viridis', marker='o')
                    axes[i].axis('equal')
                    # axes[i].set_xlabel('component 1')
                    # axes[i].set_ylabel('component 2')
                else:
                    axes[j, i].scatter(low_dimension[:, 0], low_dimension[:, 1],
                                c=sim_score, edgecolor='none', alpha=1, label=cur_class, cmap='viridis', marker='o')
                    axes[j, i].axis('equal')
                    # axes[j, i].set_xlabel('component 1')
                    # axes[j, i].set_ylabel('component 2')

        # row titles are the object classes
        if len(dimension_reduction_algos) == 1 and len(desired_classes) == 1:
            axes.set_ylabel(desired_classes[0], rotation=0, size='x-large')
            axes.set_title(dimension_reduction_algos[0], rotation=0, size='x-large')

        elif len(dimension_reduction_algos) == 1:
            for ax, row in zip(axes, desired_classes):
                ax.set_ylabel(row, rotation=0, size='x-large')
                # cbar = plt.colorbar(a)
                # cbar.set_label(f'{similarity}')
            for ax, col in zip(axes, dimension_reduction_algos):
                ax.set_title(col, rotation=0, size='x-large')

        elif len(desired_classes) == 1:
            for ax, col in zip(axes, dimension_reduction_algos):
                ax.set_title(col, rotation=0, size='x-large')
        else:
            for ax, row in zip(axes[:, 0], desired_classes):
                ax.set_ylabel(row, rotation=0, size='x-large')
            # col titles are the model variants
            for ax, col in zip(axes[0], dimension_reduction_algos):
                ax.set_title(col, rotation=0, size='x-large')

        fig.suptitle(f'Dimension reduction vis on {quantity} with {similarity} similarity score', size='xx-large')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        output_path = f'/home/zhuokai/Desktop/UChicago/Research/Visualizing-equivariant-properties/detection/faster_rcnn/figs/dimension_reduction_vis_{quantity}_{similarity}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f'\nDimension reduction vis on {quantity} with {similarity} similarity has been saved to {output_path}\n')
