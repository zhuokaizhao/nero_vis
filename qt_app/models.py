import torch
import e2cnn
import numpy as np
import torchvision
from scipy import signal
from e2cnn import gspaces
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from scipy.ndimage.filters import convolve as filter2
from typing import Optional, Tuple, List, Union
from collections import OrderedDict

from correlation import FunctionCorrelation  # the custom cost volume layer

from numpy import ma
from scipy.fft import rfft2 as rfft2_, irfft2 as irfft2_, fftshift as fftshift_
from numpy import log

######################################## Digit Recognition (MNIST) ########################################
# non-rotation equivariant network
class Non_Eqv_Net_MNIST(torch.nn.Module):
    def __init__(self, type, n_classes=10):
        super(Non_Eqv_Net_MNIST, self).__init__()
        # type either shift or rotation, matters in the length of fc layers
        self.type = type
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=24,
                            kernel_size=7,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=24),
            torch.nn.ReLU(inplace=True)
        )

        # convolution 2
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=24,
                            out_channels=48,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=48),
            torch.nn.ReLU(inplace=True)
        )

        self.pool1 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # convolution 3
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=48,
                            out_channels=48,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=48),
            torch.nn.ReLU(inplace=True)
        )

        # convolution 4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=48,
                            out_channels=96,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=96),
            torch.nn.ReLU(inplace=True)
        )

        self.pool2 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # convolution 5
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96,
                            out_channels=96,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=96),
            torch.nn.ReLU(inplace=True)
        )

        # convolution 6
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96,
                            out_channels=64,
                            kernel_size=5,
                            stride=1,
                            padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True)
        )

        self.pool3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=1)
        )

        # fully connected
        if self.type == 'rotation':
            self.fully_net = torch.nn.Sequential(
                torch.nn.Linear(576, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(64, n_classes),
            )
        elif self.type == 'shift':
            self.fully_net = torch.nn.Sequential(
                torch.nn.Linear(10816, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(64, n_classes),
            )
        elif self.type == 'scale':
            self.fully_net = torch.nn.Sequential(
                # torch.nn.Linear(6400, 64),
                torch.nn.Linear(4096, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(64, n_classes),
            )

        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)
        # pool over the spatial dimensions
        x = self.pool3(x)

        # classify with the final fully connected layers)
        # print(x.reshape(x.shape[0], -1).shape)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        # softmax to produce probability
        x = self.softmax(x)

        return x


# rotation equivariant network using e2cnn
class Rot_Eqv_Net_MNIST(torch.nn.Module):
    def __init__(self, image_size, num_rotation, n_classes=10):

        super(Rot_Eqv_Net_MNIST, self).__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=num_rotation)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2cnn.nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        if image_size != None:
            self.block1 = e2cnn.nn.SequentialModule(
                e2cnn.nn.MaskModule(in_type, image_size[0], margin=image_size[0]-28),
                e2cnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=2, bias=False),
                e2cnn.nn.InnerBatchNorm(out_type),
                e2cnn.nn.ReLU(out_type, inplace=True)
            )
        else:
            self.block1 = e2cnn.nn.SequentialModule(
                e2cnn.nn.MaskModule(in_type, 29, margin=1),
                e2cnn.nn.R2Conv(in_type, out_type, kernel_size=7, padding=2, bias=False),
                e2cnn.nn.InnerBatchNorm(out_type),
                e2cnn.nn.ReLU(out_type, inplace=True)
            )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2cnn.nn.SequentialModule(
            e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = e2cnn.nn.SequentialModule(
            e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = e2cnn.nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = e2cnn.nn.SequentialModule(
            e2cnn.nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            e2cnn.nn.InnerBatchNorm(out_type),
            e2cnn.nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = e2cnn.nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)

        self.gpool = e2cnn.nn.GroupPooling(out_type)

        # number of output channels
        c = self.gpool.out_type.size
        # c = 6400

        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2cnn.nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.block5(x)
        x = self.block6(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        # softmax to produce probability
        x = self.softmax(x)

        return x



######################################## Object Detection (COCO) ########################################
# Custom-trained model with different levels of jittering
class Custom_Trained_FastRCNN(torch.nn.Module):
    def __init__(self, num_classes=5, image_size=128):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                            num_classes=num_classes+1,
                                                                            pretrained_backbone=True,
                                                                            min_size=image_size)
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    def forward(self, x):
        x = self.model(x)

        return x



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
						input=FunctionCorrelation(tensorFirst=feat1,
												  tensorSecond=feat2,
												  intStride=1),
						negative_slope=0.1, inplace=False)
				else:  # Upsampling the correlation result (for lower resolution level)
					corr_M_out = self.upCorr_M(torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tensorFirst=feat1,
												  tensorSecond=feat2,
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
						input=FunctionCorrelation(tensorFirst=feat1,
												  tensorSecond=feat2,
												  intStride=1),
						negative_slope=0.1, inplace=False)
				else:  # Upsampling the correlation result (for lower resolution level)
					corr_M_out = self.upCorr_M(torch.nn.functional.leaky_relu(
						input=FunctionCorrelation(tensorFirst=feat1,
												  tensorSecond=feat2,
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


# Horn-Schunck method
def Horn_Schunck(im1, im2, alpha=1, Niter=100):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    def computeDerivatives(im1, im2):
        # build kernels for calculating derivatives
        kernelX = np.array([[-1, 1],
                            [-1, 1]]) * .25 #kernel for computing d/dx
        kernelY = np.array([[-1,-1],
                            [ 1, 1]]) * .25 #kernel for computing d/dy
        kernelT = np.ones((2,2))*.25

        fx = filter2(im1,kernelX) + filter2(im2,kernelX)
        fy = filter2(im1,kernelY) + filter2(im2,kernelY)

        #ft = im2 - im1
        ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

        return fx,fy,ft

	#set up initial velocities
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])

	# Set initial value for the flow vectors
    U = uInitial
    V = vInitial

	# Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

	# Averaging kernel
    kernel=np.array([[1/12, 1/6, 1/12],
                      [1/6,    0, 1/6],
                      [1/12, 1/6, 1/12]],float)

	# Iteration to reduce error
    for _ in range(Niter):
        # Compute local averages of the flow vectors
        uAvg = filter2(U,kernel)
        vAvg = filter2(V,kernel)
        # common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
        # iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U,V


# Lucas-Kanade method
def Lucas_Kanade(I1g, I2g, window_size=7, tau=1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(window_size/2) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    # I1g = I1g / 255. # normalize pixels
    # I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + \
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here

            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                u[i,j]=nu[0]
                v[i,j]=nu[1]

    return (u, v)
