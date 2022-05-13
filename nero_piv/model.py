# model file of PIV-LiteFlowNet-en
# Author: Zhuokai Zhao

import torch
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, List, Union

import correlation
import layers
from correlation import FunctionCorrelation  # the custom cost volume layer

# use cudnn
torch.backends.cudnn.enabled = True

# PIV-LiteFlowNet-En (Cai)
class PIV_LiteFlowNet_en(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PIV_LiteFlowNet_en, self).__init__()

        # number of channels for input images
        self.num_channels = 1
        # magnitude change parameter of the flow
        self.backward_scale = [0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625]

        # feature descriptor (generate pyramidal features)
        class NetC(torch.nn.Module):
            def __init__(self, num_channels):
                super(NetC, self).__init__()

                # six-module (levels) feature extractor
                self.module_one = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=num_channels,
                                    out_channels=32,
                                    kernel_size=7,
                                    stride=1,
                                    padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_two = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # third conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_three = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_four = torch.nn.Sequential(
                    # first conv + relu
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    # second conv + relu
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=96,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_five = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=96,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.module_six = torch.nn.Sequential(
                    # 'SAME' padding
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=192,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, image_tensor):
                # generate six level features
                level_one_feat = self.module_one(image_tensor)
                level_two_feat = self.module_two(level_one_feat)
                level_three_feat = self.module_three(level_two_feat)
                level_four_feat = self.module_four(level_three_feat)
                level_five_feat = self.module_five(level_four_feat)
                level_six_feat = self.module_six(level_five_feat)

                return [level_one_feat,
                        level_two_feat,
                        level_three_feat,
                        level_four_feat,
                        level_five_feat,
                        level_six_feat]

        # matching unit
        class Matching(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Matching, self).__init__()

                self.flow_scale = backward_scale[level]

                if level == 2:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                else:
                    self.feature_net = torch.nn.Sequential()

                # No flow at the top level so no need to upsample
                if level == 6:
                    self.upsample_flow = None
                # up-sample the flow
                else:
                    self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=2,
                                                                  out_channels=2,
                                                                  kernel_size=4,
                                                                  stride=2,
                                                                  padding=1,
                                                                  bias=False,
                                                                  groups=2)

                # to speed up, no correlation on level 4, 5, 6
                if level >= 4:
                    self.upsample_corr = None
                # upsample the correlation
                else:
                    self.upsample_corr = torch.nn.ConvTranspose2d(in_channels=49,
                                                                  out_channels=49,
                                                                  kernel_size=4,
                                                                  stride=2,
                                                                  padding=1,
                                                                  bias=False,
                                                                  groups=49)

                self.matching_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                    stride=1,
                                    padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                )


            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # process feature tensors further based on levels
                feat_tensor_1 = self.feature_net(feat_tensor_1)
                feat_tensor_2 = self.feature_net(feat_tensor_2)

                # upsample and scale the current flow
                if self.upsample_flow != None:
                    flow_tensor = self.upsample_flow(flow_tensor)
                    flow_tensor_scaled = flow_tensor * self.flow_scale
                    # feature warping
                    feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)

                # level 4, 5, 6 it is None
                if self.upsample_corr == None:
                    # compute the corelation between feature 1 and warped feature 2
                    corr_tensor = correlation.FunctionCorrelation(tenFirst=feat_tensor_1, tenSecond=feat_tensor_2, intStride=1)
                    corr_tensor = torch.nn.functional.leaky_relu(input=corr_tensor, negative_slope=0.1, inplace=False)
                else:
                    # compute the corelation between feature 1 and warped feature 2
                    corr_tensor = correlation.FunctionCorrelation(tenFirst=feat_tensor_1, tenSecond=feat_tensor_2, intStride=2)
                    corr_tensor = torch.nn.functional.leaky_relu(input=corr_tensor, negative_slope=0.1, inplace=False)
                    corr_tensor = self.upsample_corr(corr_tensor)

                # put correlation into matching CNN
                delta_um = self.matching_cnn(corr_tensor)

                if flow_tensor != None:
                    return flow_tensor + delta_um
                else:
                    return delta_um

        # subpixel unit
        class Subpixel(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Subpixel, self).__init__()

                self.flow_scale = backward_scale[level]

                # same feature process as in Matching
                if level == 2:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=64,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )
                    # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                    self.normalize_flow = torch.nn.Conv2d(in_channels=2,
                                                            out_channels=64,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)
                else:
                    self.feature_net = torch.nn.Sequential()
                    # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                    self.normalize_flow = torch.nn.Conv2d(in_channels=2,
                                                            out_channels=32,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1)


                # subpixel CNN that trains output to further improve flow accuracy
                self.subpixel_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 192, 160, 224, 288, 416 ][level],
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=2,
                                    kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                    stride=1,
                                    padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                )




            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # process feature tensors further based on levels
                feat_tensor_1 = self.feature_net(feat_tensor_1)
                feat_tensor_2 = self.feature_net(feat_tensor_2)

                if flow_tensor != None:
                    # use flow from matching unit to warp feature 2 again
                    flow_tensor_scaled = flow_tensor * self.flow_scale
                    feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)

                # PIV-LiteFlowNet-en change 2, make the dimensionality of velocity field consistent with image features
                flow_tensor_scaled = self.normalize_flow(flow_tensor_scaled)

                # volume that is going to be fed into subpixel CNN
                volume = torch.cat([feat_tensor_1, feat_tensor_2, flow_tensor_scaled], axis=1)
                delta_us = self.subpixel_cnn(volume)

                return flow_tensor + delta_us

        class Regularization(torch.nn.Module):
            def __init__(self, level, backward_scale):
                super(Regularization, self).__init__()

                self.flow_scale = backward_scale[level]

                self.unfold = [ 0, 0, 7, 5, 5, 3, 3 ][level]

                if level >= 5:
                    self.feature_net = torch.nn.Sequential()

                    self.dist_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][level],
                                        stride=1,
                                        padding=[ 0, 0, 3, 2, 2, 1, 1 ][level])
                    )
                else:
                    self.feature_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][level],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                    self.dist_net = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32,
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][level], 1),
                                        stride=1,
                                        padding=([ 0, 0, 3, 2, 2, 1, 1 ][level], 0)),

                        torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                        kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][level]),
                                        stride=1,
                                        padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][level]))
                    )


                # network that scales x and y
                self.scale_x_net = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                                    out_channels=1,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

                self.scale_y_net = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][level],
                                                    out_channels=1,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

                self.regularization_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][level],
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),

                    torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )


            def forward(self, flow_tensor, image_tensor_1, image_tensor_2, feat_tensor_1, feat_tensor_2):

                # distance between feature 1 and warped feature 2
                flow_tensor_scaled = flow_tensor * self.flow_scale
                feat_tensor_2 = layers.backwarp(feat_tensor_2, flow_tensor_scaled)
                squared_diff_tensor = torch.pow((feat_tensor_1 - feat_tensor_2), 2)
                # sum the difference in both x and y
                squared_diff_tensor = torch.sum(squared_diff_tensor, dim=1, keepdim=True)
                # take the square root
                # diff_tensor = torch.sqrt(squared_diff_tensor)
                diff_tensor = squared_diff_tensor

                # construct volume
                volume_tensor = torch.cat([ diff_tensor, flow_tensor - flow_tensor.view(flow_tensor.shape[0], 2, -1).mean(2, True).view(flow_tensor.shape[0], 2, 1, 1), self.feature_net(feat_tensor_1) ], axis=1)
                dist_tensor = self.regularization_cnn(volume_tensor)
                dist_tensor = self.dist_net(dist_tensor)
                dist_tensor = dist_tensor.pow(2.0).neg()
                dist_tensor = (dist_tensor - dist_tensor.max(1, True)[0]).exp()

                divisor_tensor = dist_tensor.sum(1, True).reciprocal()

                dist_tensor_unfold_x = torch.nn.functional.unfold(input=flow_tensor[:, 0:1, :, :],
                                                                  kernel_size=self.unfold,
                                                                  stride=1,
                                                                  padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)

                dist_tensor_unfold_y = torch.nn.functional.unfold(input=flow_tensor[:, 1:2, :, :],
                                                                  kernel_size=self.unfold,
                                                                  stride=1,
                                                                  padding=int((self.unfold - 1) / 2)).view_as(dist_tensor)


                scale_x_tensor = divisor_tensor * self.scale_x_net(dist_tensor * dist_tensor_unfold_x)
                scale_y_tensor = divisor_tensor * self.scale_y_net(dist_tensor * dist_tensor_unfold_y)


                return torch.cat([scale_x_tensor, scale_y_tensor], axis=1)

        # combine all units
        self.NetC = NetC(self.num_channels)
        self.Matching = torch.nn.ModuleList([Matching(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])
        self.Subpixel = torch.nn.ModuleList([Subpixel(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])
        self.Regularization = torch.nn.ModuleList([Regularization(level, self.backward_scale) for level in [2, 3, 4, 5, 6]])

        self.upsample_flow = torch.nn.ConvTranspose2d(in_channels=2,
                                                        out_channels=2,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=1,
                                                        bias=False,
                                                        groups=2)

    def forward(self, input_image_pair):
        image_tensor_1 = input_image_pair[:, 0:1, :, :]
        image_tensor_2 = input_image_pair[:, 1:, :, :]
        feat_tensor_pyramid_1 = self.NetC(image_tensor_1)
        feat_tensor_pyramid_2 = self.NetC(image_tensor_2)

        image_tensor_pyramid_1 = [image_tensor_1]
        image_tensor_pyramid_2 = [image_tensor_2]
        for level in [1, 2, 3, 4, 5]:
            # downsample image to match the different levels in the feature pyramid
            new_image_tensor_1 = torch.nn.functional.interpolate(input=image_tensor_pyramid_1[-1],
                                                                 size=(feat_tensor_pyramid_1[level].shape[2], feat_tensor_pyramid_1[level].shape[3]),
                                                                 mode='bilinear',
                                                                 align_corners=False)

            image_tensor_pyramid_1.append(new_image_tensor_1)

            new_image_tensor_2 = torch.nn.functional.interpolate(input=image_tensor_pyramid_2[-1],
                                                                 size=(feat_tensor_pyramid_2[level].shape[2], feat_tensor_pyramid_2[level].shape[3]),
                                                                 mode='bilinear',
                                                                 align_corners=False)

            image_tensor_pyramid_2.append(new_image_tensor_2)


        # initialize empty flow
        flow_tensor = None

        for level in [-1, -2, -3, -4, -5]:
            flow_tensor = self.Matching[level](flow_tensor,
                                               image_tensor_pyramid_1[level],
                                               image_tensor_pyramid_2[level],
                                               feat_tensor_pyramid_1[level],
                                               feat_tensor_pyramid_2[level])

            flow_tensor = self.Subpixel[level](flow_tensor,
                                               image_tensor_pyramid_1[level],
                                               image_tensor_pyramid_2[level],
                                               feat_tensor_pyramid_1[level],
                                               feat_tensor_pyramid_2[level])

            flow_tensor = self.Regularization[level](flow_tensor,
                                                     image_tensor_pyramid_1[level],
                                                     image_tensor_pyramid_2[level],
                                                     feat_tensor_pyramid_1[level],
                                                     feat_tensor_pyramid_2[level])

        # upsample flow tensor
        # flow_tensor = torch.nn.functional.interpolate(input=flow_tensor,
        #                                                 size=(flow_tensor.shape[2]*2, flow_tensor.shape[3]*2),
        #                                                 mode='bilinear',
        #                                                 align_corners=False)

        # PIV-LiteFlowNet-en uses deconv to upsample (change 1)
        flow_tensor = self.upsample_flow(flow_tensor)

        return flow_tensor


######################################## PIV ########################################
# Below code are taken from https://github.com/abrosua/piv_liteflownet-pytorch/tree/04b73b4cd72c916b7f23afbed99bc015753151da
#################################      NOTE      #################################
#	1. ConvTranspose2d is the equivalent of Deconvolution layer in Caffe
# 		(it's NOT an UPSAMPLING layer, since it's trainable!)
#
#
##################################################################################

backwarp_tensorGrid = {}


def backwarp(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in backwarp_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.shape[3]).view(1, 1, 1, tensorFlow.shape[3]).expand(
			tensorFlow.shape[0], -1, tensorFlow.shape[2], -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.shape[2]).view(1, 1, tensorFlow.shape[2], 1).expand(
			tensorFlow.shape[0], -1, -1, tensorFlow.shape[3])

		backwarp_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()

	tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.shape[3] - 1.0) / 2.0),
							tensorFlow[:, 1:2, :, :] / ((tensorInput.shape[2] - 1.0) / 2.0)], 1)

	return torch.nn.functional.grid_sample(input=tensorInput,
										   grid=(backwarp_tensorGrid[str(tensorFlow.size())] +
												 tensorFlow).permute(0, 2, 3, 1),
										   mode='bilinear', padding_mode='zeros', align_corners=True)


# ----------- NETWORK DEFINITION -----------
class LiteFlowNet(torch.nn.Module):
	def __init__(self, starting_scale: int = 40, lowest_level: int = 2,
				 rgb_mean: Union[Tuple[float, ...], List[float]] = (0.411618, 0.434631, 0.454253, 0.410782, 0.433645, 0.452793)
				 ) -> None:
		"""
		LiteFlowNet network architecture by Hui, 2018.
		The default rgb_mean value is obtained from the original Caffe model.
		:param starting_scale: Flow factor value for the first pyramid layer.
		:param lowest_level: Determine the last pyramid level to use for the decoding.
		:param rgb_mean: The datasets mean pixel rgb value.
		"""
		super(LiteFlowNet, self).__init__()
		self.mean_aug = [[0, 0, 0], [0, 0, 0]]

		## INIT.
		rgb_mean = list(rgb_mean)
		self.MEAN = [rgb_mean[:3], rgb_mean[3:]]
		self.lowest_level = int(lowest_level)
		self.PLEVELS = 6
		self.SCALEFACTOR = []

		# NEGLECT the first index (Level 0)
		for level in range(self.PLEVELS + 1):
			FACTOR = 2.0 ** level
			self.SCALEFACTOR.append(float(starting_scale) / FACTOR)

		## NetC: Pyramid feature extractor
		class Features(torch.nn.Module):
			def __init__(self):
				super(Features, self).__init__()

				self.conv1 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv2 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv3 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv4 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv5 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv6 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, tensor_input):
				level1_feat = self.conv1(tensor_input)
				level2_feat = self.conv2(level1_feat)
				level3_feat = self.conv3(level2_feat)
				level4_feat = self.conv4(level3_feat)
				level5_feat = self.conv5(level4_feat)
				level6_feat = self.conv6(level5_feat)

				return [level1_feat, level2_feat, level3_feat, level4_feat, level5_feat, level6_feat]

		## NetC_ext: Feature matching extension for pyramid level 2 (and 1) only
		class FeatureExt(torch.nn.Module):
			def __init__(self):
				super(FeatureExt, self).__init__()

				self.conv_ext = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, feat):
				feat_out = self.conv_ext(feat)

				return feat_out

		## NetE: Descriptor matching (M)
		class Matching(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Matching, self).__init__()

				self.fltBackwarp = scale_factor

				# Level 6 modifier
				if pyr_level == 6:
					self.upConv_M = None
				else:  # upsampling with TRAINABLE parameters
					self.upConv_M = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4,
															 stride=2, padding=1, bias=False, groups=2)

				# Level 6 to 4 modifier
				if pyr_level >= 4:
					self.upCorr_M = None
				else:  # Upsampling the correlation result for level with lower resolution
					self.upCorr_M = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4,
															 stride=2, padding=1, bias=False, groups=49)

				self.conv_M = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level],
									stride=1, padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
				)

			def forward(self, tensorFirst, tensorSecond, feat1, feat2, xflow):
				# feat1 = self.moduleFeat(feat1)
				# feat2 = self.moduleFeat(feat2)

				if xflow is not None:
					xflow = self.upConv_M(xflow)  # s * (x_dot ^s)
					feat2 = backwarp(tensorInput=feat2, tensorFlow=xflow * self.fltBackwarp)  # backward warping

				if self.upCorr_M is None:
					corr_M_out = torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tenFirst=feat1,
												  tenSecond=feat2,
												  intStride=1),
						negative_slope=0.1, inplace=False)
				else:  # Upsampling the correlation result (for lower resolution level)
					corr_M_out = self.upCorr_M(torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tenFirst=feat1,
												  tenSecond=feat2,
												  intStride=2),
						negative_slope=0.1, inplace=False))

				flow_M = self.conv_M(corr_M_out) + (xflow if xflow is not None else 0.0)
				return flow_M

		## NetE: Subpixel refinement (S)
		class Subpixel(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Subpixel, self).__init__()

				# scalar multiplication to upsample the flow (s * x_dot ** s)
				self.fltBackward = scale_factor

				self.conv_S = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[0, 130, 130, 130, 194, 258, 386][pyr_level],
									out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level],
									stride=1, padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
				)

			def forward(self, img1, img2, feat1, feat2, xflow):
				# feat1 = self.moduleFeat(feat1)
				# feat2 = self.moduleFeat(feat2)

				if xflow is not None:  # at this point xflow is already = s * (x_dot ^s) due to upconv/ConvTrans2d in M
					feat2 = backwarp(tensorInput=feat2, tensorFlow=xflow * self.fltBackward)

				flow_S = self.conv_S(torch.cat([feat1, feat2, xflow], 1)) + (xflow if xflow is not None else 0.0)
				return flow_S

		## NetE: Flow Regularization (R)
		class Regularization(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Regularization, self).__init__()

				self.fltBackward = scale_factor
				self.intUnfold = [0, 7, 7, 5, 5, 3, 3][pyr_level]

				if pyr_level < 5:
					self.moduleFeat = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=[0, 32, 32, 64, 96, 128, 192][pyr_level],
										out_channels=128, kernel_size=1, stride=1, padding=0),
						torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
					)
				else:  # pyramid level 5 and 6 only!
					self.moduleFeat = torch.nn.Sequential()

				self.conv_R = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[0, 131, 131, 131, 131, 131, 195][pyr_level],
									out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				if pyr_level < 5:
					self.conv_dist_R = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=([0, 7, 7, 5, 5, 3, 3][pyr_level], 1),
										stride=1, padding=([0, 3, 3, 2, 2, 1, 1][pyr_level], 0)),
						torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=(1, [0, 7, 7, 5, 5, 3, 3][pyr_level]),
										stride=1, padding=(0, [0, 3, 3, 2, 2, 1, 1][pyr_level]))
					)
				else:  # pyramid level 5 and 6 only!
					self.conv_dist_R = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level], stride=1,
										padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
					)

				self.moduleScaleX = torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
													out_channels=1, kernel_size=1, stride=1, padding=0)
				self.moduleScaleY = torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
													out_channels=1, kernel_size=1, stride=1, padding=0)

			def forward(self, img1, img2, feat1, feat2, xflow_S):
				rm_flow_R = xflow_S - xflow_S.view(xflow_S.shape[0], 2, -1).mean(2, True).view(xflow_S.shape[0], 2, 1, 1)
				rgb_warp_R = backwarp(tensorInput=img2, tensorFlow=xflow_S * self.fltBackward)  # ensure the same shape!
				norm_R = (img1 - rgb_warp_R).pow(2.0).sum(1, True).sqrt().detach()  # L2 norm operation

				conv_dist_R_out = self.conv_dist_R(
					self.conv_R(torch.cat([norm_R, rm_flow_R, self.moduleFeat(feat1)], 1)))
				negsq_R = conv_dist_R_out.pow(2.0).neg()  # Negative-square

				# Softmax
				tensorDist = (negsq_R - negsq_R.max(1, True)[0]).exp()  #
				tensorDivisor = tensorDist.sum(1, True).reciprocal()  # output_{i,j,k} = 1 / input_{i,j,k}

				# Element-wise dot product (.)
				xflow_scale = self.moduleScaleX(
					tensorDist *
					torch.nn.functional.unfold(
						input=xflow_S[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1,
						padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)
				) * tensorDivisor

				yflow_scale = self.moduleScaleY(
					tensorDist *
					torch.nn.functional.unfold(
						input=xflow_S[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1,
						padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)
				) * tensorDivisor

				flow_R = torch.cat([xflow_scale, yflow_scale], 1)
				return flow_R

		## Back to the main model Class
		self.level2use = list(range(self.lowest_level, self.PLEVELS + 1))  # Define which layers to use
		self.NetC = Features()  # NetC - Feature extractor
		# Extra feature extractor
		self.NetC_ext = torch.nn.ModuleList([
			FeatureExt() for i in range(self.lowest_level - 1, 2)
		])
		self.NetE_M = torch.nn.ModuleList([
			Matching(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - M
		self.NetE_S = torch.nn.ModuleList([
			Subpixel(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - S
		self.NetE_R = torch.nn.ModuleList([
			Regularization(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - R

	def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
		# Mean normalization due to training augmentation
		for i in range(img1.shape[1]):
			img1[:, i, :, :] = img1[:, i, :, :] - self.MEAN[0][i]
			img2[:, i, :, :] = img2[:, i, :, :] - self.MEAN[1][i]

		feat1 = self.NetC(img1)
		feat2 = self.NetC(img2)

		img1 = [img1]
		img2 = [img2]

		# Iteration init.
		len_level, len_feat = len(self.level2use), len(feat1)
		idx_diff = int(np.abs(len_feat - len_level))

		## NetC: Interpolate image to the feature size from the level 2 up to level 6
		for pyr_level in range(1, len_feat):
			# using bilinear interpolation
			img1.append(torch.nn.functional.interpolate(input=img1[-1],
														size=(feat1[pyr_level].shape[2], feat1[pyr_level].shape[3]),
														mode='bilinear', align_corners=False))
			img2.append(torch.nn.functional.interpolate(input=img2[-1],
														size=(feat2[pyr_level].shape[2], feat2[pyr_level].shape[3]),
														mode='bilinear', align_corners=False))

		## NetE: Stacking the NetE network from Level 6 up to the LOWEST level (1 or 2)
		# variable init.
		xflow = None
		xflow_train = []

		for i_level in reversed(range(0, len_level)):
			pyr_level = i_level + idx_diff

			if pyr_level < 2:
				f1_in = self.NetC_ext[pyr_level - 1](feat1[pyr_level])
				f2_in = self.NetC_ext[pyr_level - 1](feat2[pyr_level])
			else:
				f1_in, f2_in = feat1[pyr_level], feat2[pyr_level]

			xflow_M = self.NetE_M[i_level](img1[pyr_level], img2[pyr_level], f1_in, f2_in, xflow)
			xflow_S = self.NetE_S[i_level](img1[pyr_level], img2[pyr_level], f1_in, f2_in, xflow_M)
			xflow = self.NetE_R[i_level](img1[pyr_level], img2[pyr_level], feat1[pyr_level], feat2[pyr_level], xflow_S)

			xflow_train.append([xflow_M, xflow_S, xflow])

		if self.training:  # training mode
			# return [xf_factor * (self.SCALEFACTOR[1]) for xf_factor in xflow_train]
			return xflow_train

		else:  # evaluation mode
			return xflow * (self.SCALEFACTOR[1])  # FlowNet's practice (use the Level 1's scale factor)


class LiteFlowNet2(torch.nn.Module):
	def __init__(self, starting_scale: int = 40, lowest_level: int = 3,
				 rgb_mean: Union[Tuple[float, ...], List[float]] = (0.411618, 0.434631, 0.454253, 0.410782, 0.433645, 0.452793)
				 ) -> None:
		"""
		LiteFlowNet2 network architecture by Hui, 2020.
		The default rgb_mean value is obtained from the original Caffe model.
		:param starting_scale: Flow factor value for the first pyramid layer.
		:param lowest_level: Determine the last pyramid level to use for the decoding.
		:param rgb_mean: The datasets mean pixel rgb value.
		"""
		super(LiteFlowNet2, self).__init__()

		## INIT.
		rgb_mean = list(rgb_mean)
		self.MEAN = [rgb_mean[:3], rgb_mean[3:]]
		self.lowest_level = int(lowest_level)
		self.PLEVELS = 6
		self.SCALEFACTOR = []

		# NEGLECT the first index (Level 0)
		for level in range(self.PLEVELS + 1):
			FACTOR = 2.0 ** level
			self.SCALEFACTOR.append(float(starting_scale) / FACTOR)

		## NetC: Pyramid feature extractor
		class Features(torch.nn.Module):
			def __init__(self):
				super(Features, self).__init__()

				self.conv1 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv2 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv3 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv4 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv5 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.conv6 = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, tensor_input):
				level1_feat = self.conv1(tensor_input)
				level2_feat = self.conv2(level1_feat)
				level3_feat = self.conv3(level2_feat)
				level4_feat = self.conv4(level3_feat)
				level5_feat = self.conv5(level4_feat)
				level6_feat = self.conv6(level5_feat)

				return [level1_feat, level2_feat, level3_feat, level4_feat, level5_feat, level6_feat]

		## NetC_ext: Feature matching extension for pyramid level 2 (and 1) only
		class FeatureExt(torch.nn.Module):
			def __init__(self):
				super(FeatureExt, self).__init__()

				self.conv_ext = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, feat):
				feat_out = self.conv_ext(feat)

				return feat_out

		## NetE: Descriptor matching (M)
		class Matching(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Matching, self).__init__()

				self.fltBackwarp = scale_factor

				# Level 6 modifier
				if pyr_level == 6:
					self.upConv_M = None
				else:  # upsampling with TRAINABLE parameters
					self.upConv_M = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4,
															 stride=2, padding=1, bias=False, groups=2)

				# Level 6 to 4 modifier
				if pyr_level >= 4:
					self.upCorr_M = None
				else:  # Upsampling the correlation result for level with lower resolution
					self.upCorr_M = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4,
															 stride=2, padding=1, bias=False, groups=49)

				self.conv_M = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level],
									stride=1, padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
				)

			def forward(self, tensorFirst, tensorSecond, feat1, feat2, xflow):
				# feat1 = self.moduleFeat(feat1)
				# feat2 = self.moduleFeat(feat2)

				if xflow is not None:
					xflow = self.upConv_M(xflow)  # s * (x_dot ^s)
					feat2 = backwarp(tensorInput=feat2, tensorFlow=xflow * self.fltBackwarp)  # backward warping

				if self.upCorr_M is None:
					corr_M_out = torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tenFirst=feat1,
												  tenSecond=feat2,
												  intStride=1),
						negative_slope=0.1, inplace=False)
				else:  # Upsampling the correlation result (for lower resolution level)
					corr_M_out = self.upCorr_M(torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tenFirst=feat1,
												  tenSecond=feat2,
												  intStride=2),
						negative_slope=0.1, inplace=False))

				flow_M = self.conv_M(corr_M_out) + (xflow if xflow is not None else 0.0)
				return flow_M

		## NetE: Subpixel refinement (S)
		class Subpixel(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Subpixel, self).__init__()

				# scalar multiplication to upsample the flow (s * x_dot ** s)
				self.fltBackward = scale_factor

				self.conv_S = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[0, 130, 130, 130, 194, 258, 386][pyr_level],
									out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level],
									stride=1, padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
				)

			def forward(self, img1, img2, feat1, feat2, xflow):
				# feat1 = self.moduleFeat(feat1)
				# feat2 = self.moduleFeat(feat2)

				if xflow is not None:  # at this point xflow is already = s * (x_dot ^s) due to upconv/ConvTrans2d in M
					feat2 = backwarp(tensorInput=feat2, tensorFlow=xflow * self.fltBackward)

				flow_S = self.conv_S(torch.cat([feat1, feat2, xflow], 1)) + (xflow if xflow is not None else 0.0)
				return flow_S

		## NetE: Flow Regularization (R)
		class Regularization(torch.nn.Module):
			def __init__(self, pyr_level, scale_factor):
				super(Regularization, self).__init__()

				self.fltBackward = scale_factor
				self.intUnfold = [0, 7, 7, 5, 5, 3, 3][pyr_level]

				if pyr_level < 5:
					self.moduleFeat = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=[0, 32, 32, 64, 96, 128, 192][pyr_level],
										out_channels=128, kernel_size=1, stride=1, padding=0),
						torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
					)
				else:  # pyramid level 5 and 6 only!
					self.moduleFeat = torch.nn.Sequential()

				self.conv_R = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=[0, 131, 131, 131, 131, 131, 195][pyr_level],
									out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				if pyr_level < 5:
					self.conv_dist_R = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=([0, 7, 7, 5, 5, 3, 3][pyr_level], 1),
										stride=1, padding=([0, 3, 3, 2, 2, 1, 1][pyr_level], 0)),
						torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=(1, [0, 7, 7, 5, 5, 3, 3][pyr_level]),
										stride=1, padding=(0, [0, 3, 3, 2, 2, 1, 1][pyr_level]))
					)
				else:  # pyramid level 5 and 6 only!
					self.conv_dist_R = torch.nn.Sequential(
						torch.nn.Conv2d(in_channels=32, out_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
										kernel_size=[0, 7, 7, 5, 5, 3, 3][pyr_level], stride=1,
										padding=[0, 3, 3, 2, 2, 1, 1][pyr_level])
					)

				self.moduleScaleX = torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
													out_channels=1, kernel_size=1, stride=1, padding=0)
				self.moduleScaleY = torch.nn.Conv2d(in_channels=[0, 49, 49, 25, 25, 9, 9][pyr_level],
													out_channels=1, kernel_size=1, stride=1, padding=0)

			def forward(self, img1, img2, feat1, feat2, xflow_S):
				rm_flow_R = xflow_S - xflow_S.view(xflow_S.shape[0], 2, -1).mean(2, True).view(xflow_S.shape[0], 2, 1, 1)
				rgb_warp_R = backwarp(tensorInput=img2, tensorFlow=xflow_S * self.fltBackward)  # ensure the same shape!
				norm_R = (img1 - rgb_warp_R).pow(2.0).sum(1, True).sqrt().detach()  # L2 norm operation

				conv_dist_R_out = self.conv_dist_R(
					self.conv_R(torch.cat([norm_R, rm_flow_R, self.moduleFeat(feat1)], 1)))
				negsq_R = conv_dist_R_out.pow(2.0).neg()  # Negative-square

				# Softmax
				tensorDist = (negsq_R - negsq_R.max(1, True)[0]).exp()  #
				tensorDivisor = tensorDist.sum(1, True).reciprocal()  # output_{i,j,k} = 1 / input_{i,j,k}

				# Element-wise dot product (.)
				xflow_scale = self.moduleScaleX(
					tensorDist *
					torch.nn.functional.unfold(
						input=xflow_S[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1,
						padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)
				) * tensorDivisor

				yflow_scale = self.moduleScaleY(
					tensorDist *
					torch.nn.functional.unfold(
						input=xflow_S[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1,
						padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)
				) * tensorDivisor

				flow_R = torch.cat([xflow_scale, yflow_scale], 1)
				return flow_R

		## Back to the main model Class
		self.level2use = list(range(self.lowest_level, self.PLEVELS + 1))  # Define which layers to use
		self.NetC = Features()  # NetC - Feature extractor
		# Extra feature extractor
		self.NetC_ext = torch.nn.ModuleList([
			FeatureExt() for i in range(self.lowest_level - 1, 2)
		])
		self.NetE_M = torch.nn.ModuleList([
			Matching(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - M
		self.NetE_S = torch.nn.ModuleList([
			Subpixel(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - S
		self.NetE_R = torch.nn.ModuleList([
			Regularization(pyr_level, self.SCALEFACTOR[pyr_level]) for pyr_level in self.level2use])  # NetE - R

	def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
		# Mean normalization due to training augmentation
		for i in range(img1.shape[1]):
			img1[:, i, :, :] = img1[:, i, :, :] - self.MEAN[0][i]
			img2[:, i, :, :] = img2[:, i, :, :] - self.MEAN[1][i]

		# Init.
		im_shape = (img1.shape[2], img1.shape[3])

		feat1 = self.NetC(img1)
		feat2 = self.NetC(img2)

		img1 = [img1]
		img2 = [img2]

		# Iteration init.
		len_level, len_feat = len(self.level2use), len(feat1)
		idx_diff = int(np.abs(len_feat - len_level))

		## NetC: Interpolate image to the feature size from the level 2 up to level 6
		for pyr_level in range(1, len_feat):
			# using bilinear interpolation
			img1.append(torch.nn.functional.interpolate(input=img1[-1],
														size=(feat1[pyr_level].shape[2], feat1[pyr_level].shape[3]),
														mode='bilinear', align_corners=False))
			img2.append(torch.nn.functional.interpolate(input=img2[-1],
														size=(feat2[pyr_level].shape[2], feat2[pyr_level].shape[3]),
														mode='bilinear', align_corners=False))

		## NetE: Stacking the NetE network from Level 6 up to the LOWEST level (1 or 2)
		# variable init.
		xflow = None
		xflow_train = []

		for i_level in reversed(range(0, len_level)):
			pyr_level = i_level + idx_diff

			if pyr_level < 2:
				f1_in = self.NetC_ext[pyr_level - 1](feat1[pyr_level])
				f2_in = self.NetC_ext[pyr_level - 1](feat2[pyr_level])
			else:
				f1_in, f2_in = feat1[pyr_level], feat2[pyr_level]

			xflow_M = self.NetE_M[i_level](img1[pyr_level], img2[pyr_level], f1_in, f2_in, xflow)
			xflow_S = self.NetE_S[i_level](img1[pyr_level], img2[pyr_level], f1_in, f2_in, xflow_M)
			xflow = self.NetE_R[i_level](img1[pyr_level], img2[pyr_level], feat1[pyr_level], feat2[pyr_level], xflow_S)

			xflow_train.append([xflow_M, xflow_S, xflow])

		if self.training:  # training mode
			xflow_upsampled = torch.nn.functional.interpolate(input=xflow, size=im_shape, mode='bilinear',
															  align_corners=False)
			xflow_train.append([xflow_upsampled])
			return xflow_train

		else:  # evaluation mode
			return xflow * (self.SCALEFACTOR[1])  # FlowNet's practice (use the Level 1's scale factor)


def hui_liteflownet(params: Optional[OrderedDict] = None, version: int = 1):
	"""The original LiteFlowNet model architecture from the
	"LiteFlowNet: A Lightweight Convolutional Neural Networkfor Optical Flow Estimation" paper'
	(https://arxiv.org/abs/1805.07036)
	Args:
		params 	: pretrained weights of the network (as state dict). will create a new one if not set
		version	: to determine whether to use LiteFlowNet or LiteFlowNet2 as the model
	"""
	if version == 1:
		MEAN = (0.411618, 0.434631, 0.454253, 0.410782, 0.433645, 0.452793)  # (Hui, 2018)
		model = LiteFlowNet(rgb_mean=MEAN)
	elif version == 2:  # using LiteFlowNet2
		model = LiteFlowNet2()
	else:
		raise ValueError(f'Wrong input of model version (input = {version})! Choose between version 1 or 2 only!')

	if params is not None:
		# model.load_state_dict(data['state_dict'])
		model.load_state_dict(params)

	return model


def piv_liteflownet(params: Optional[OrderedDict] = None, version: int = 1):
	"""The PIV-LiteFlowNet-en (modification of a LiteFlowNet model) model architecture from the
	"Particle image velocimetry based on a deep learning motion estimator" paper
	(https://ieeexplore.ieee.org/document/8793167)
	Args:
		params : pretrained weights of the network. will create a new one if not set
		version	: to determine whether to use LiteFlowNet or LiteFlowNet2 as the model
	"""
	# Mean augmentation global variable
	if version == 1:  # using LiteFlowNet
		MEAN = (0.173935, 0.180594, 0.192608, 0.172978, 0.179518, 0.191300)  # PIV-LiteFlowNet-en (Cai, 2019)
		model = LiteFlowNet(starting_scale=10, lowest_level=1, rgb_mean=MEAN)
	elif version == 2:  # using LiteFlowNet2
		MEAN = (0.194286, 0.190633, 0.191766, 0.194220, 0.190595, 0.191701)  # PIV-LiteFlowNet2-en (Silitonga, 2020)
		model = LiteFlowNet2(starting_scale=10, lowest_level=2, rgb_mean=MEAN)
	else:
		raise ValueError(f'Wrong input of model version (input = {version})! Choose between version 1 or 2 only!')

	if params is not None:
		# model.load_state_dict(data['state_dict'])
		model.load_state_dict(params)

	return model