import torch
import e2cnn
import numpy as np
import torchvision
from scipy import signal
from e2cnn import gspaces
import torch.utils.model_zoo as model_zoo
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


# method taken from OpenPIV
def simple_piv(im1, im2):
    """
    Simplest PIV run on the pair of images using default settings
    piv(im1,im2) will create a tmp.vec file with the vector filed in pix/dt
    (dt=1) from two images, im1,im2 provided as full path filenames
    (TIF is preferable, whatever imageio can read)
    """

    u, v, s2n = extended_search_area_piv(
        im1.astype(np.int32), im2.astype(np.int32), window_size=8,
        overlap=0, search_area_size=8
    )

    return u, v


def extended_search_area_piv(
    frame_a,
    frame_b,
    window_size,
    overlap=0,
    dt=1.0,
    search_area_size=None,
    correlation_method="circular",
    subpixel_method="gaussian",
    sig2noise_method='peak2mean',
    width=2,
    normalized_correlation=False,
    use_vectorized = False,
):
    """Standard PIV cross-correlation algorithm, with an option for
    extended area search that increased dynamic range. The search region
    in the second frame is larger than the interrogation window size in the
    first frame. For Cython implementation see
    openpiv.process.extended_search_area_piv
    This is a pure python implementation of the standard PIV cross-correlation
    algorithm. It is a zero order displacement predictor, and no iterative
    process is performed.
    Parameters
    ----------
    frame_a : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the first frame.
    frame_b : 2d np.ndarray
        an two dimensions array of integers containing grey levels of
        the second frame.
    window_size : int
        the size of the (square) interrogation window, [default: 32 pix].
    overlap : int
        the number of pixels by which two adjacent windows overlap
        [default: 16 pix].
    dt : float
        the time delay separating the two frames [default: 1.0].
    correlation_method : string
        one of the two methods implemented: 'circular' or 'linear',
        default: 'circular', it's faster, without zero-padding
        'linear' requires also normalized_correlation = True (see below)
    subpixel_method : string
         one of the following methods to estimate subpixel location of the
         peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.
    sig2noise_method : string
        defines the method of signal-to-noise-ratio measure,
        ('peak2peak' or 'peak2mean'. If None, no measure is performed.)
    width : int
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
    search_area_size : int
       the size of the interrogation window in the second frame,
       default is the same interrogation window size and it is a
       fallback to the simplest FFT based PIV
    normalized_correlation: bool
        if True, then the image intensity will be modified by removing
        the mean, dividing by the standard deviation and
        the correlation map will be normalized. It's slower but could be
        more robust
    Returns
    -------
    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.
    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.
    sig2noise : 2d np.ndarray, ( optional: only if sig2noise_method != None )
        a two dimensional array the signal to noise ratio for each
        window pair.
    The implementation of the one-step direct correlation with different
    size of the interrogation window and the search area. The increased
    size of the search areas cope with the problem of loss of pairs due
    to in-plane motion, allowing for a smaller interrogation window size,
    without increasing the number of outlier vectors.
    See:
    Particle-Imaging Techniques for Experimental Fluid Mechanics
    Annual Review of Fluid Mechanics
    Vol. 23: 261-304 (Volume publication date January 1991)
    DOI: 10.1146/annurev.fl.23.010191.001401
    originally implemented in process.pyx in Cython and converted to
    a NumPy vectorized solution in pyprocess.py
    """
    if search_area_size is not None:
        if isinstance(search_area_size, tuple) == False and isinstance(search_area_size, list) == False:
            search_area_size = [search_area_size, search_area_size]
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    # check the inputs for validity
    if search_area_size is None:
        search_area_size = window_size

    if overlap[0] >= window_size[0] or overlap[1] >= window_size[1]:
        raise ValueError("Overlap has to be smaller than the window_size")

    if search_area_size[0] < window_size[0] or search_area_size[1] < window_size[1]:
        raise ValueError("Search size cannot be smaller than the window_size")

    if (window_size[1] > frame_a.shape[0]) or (window_size[0] > frame_a.shape[1]):
        raise ValueError("window size cannot be larger than the image")

    # get field shape
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

    # We implement the new vectorized code
    aa = sliding_window_array(frame_a, search_area_size, overlap)
    bb = sliding_window_array(frame_b, search_area_size, overlap)

    # for the case of extended seearch, the window size is smaller than
    # the search_area_size. In order to keep it all vectorized the
    # approach is to use the interrogation window in both
    # frames of the same size of search_area_asize,
    # but mask out the region around
    # the interrogation window in the frame A

    if search_area_size > window_size:
        # before masking with zeros we need to remove
        # edges

        aa = normalize_intensity(aa)
        bb = normalize_intensity(bb)

        mask = np.zeros((search_area_size[0], search_area_size[1])).astype(aa.dtype)
        pady = int((search_area_size[0] - window_size[0]) / 2)
        padx = int((search_area_size[1] - window_size[1]) / 2)
        mask[slice(pady, search_area_size[0] - pady),
             slice(padx, search_area_size[1] - padx)] = 1
        mask = np.broadcast_to(mask, aa.shape)
        aa *= mask

    corr = fft_correlate_images(aa, bb,
                                correlation_method=correlation_method,
                                normalized_correlation=normalized_correlation)
    if use_vectorized == True:
        u, v = vectorized_correlation_to_displacements(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)
    else:
        u, v = correlation_to_displacement(corr, n_rows, n_cols,
                                           subpixel_method=subpixel_method)

    # return output depending if user wanted sig2noise information
    if sig2noise_method is not None:
        if use_vectorized == True:
            sig2noise = vectorized_sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
        else:
            sig2noise = sig2noise_ratio(
                corr, sig2noise_method=sig2noise_method, width=width
            )
    else:
        sig2noise = np.zeros_like(u)*np.nan

    sig2noise = sig2noise.reshape(n_rows, n_cols)

    return u/dt, v/dt, sig2noise


def sig2noise_ratio(correlation, sig2noise_method="peak2peak", width=2):
    """
    Computes the signal to noise ratio from the correlation map.
    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interrogation windows.
    Parameters
    ----------
    corr : 3d np.ndarray
        the correlation maps of the image pair, concatenated along 0th axis
    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.
    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if ``sig2noise_method==peak2peak``.
    Returns
    -------
    sig2noise : np.array
        the signal to noise ratios from the correlation maps.
    """
    sig2noise = np.zeros(correlation.shape[0])
    corr_max1 = np.zeros(correlation.shape[0])
    corr_max2 = np.zeros(correlation.shape[0])
    if sig2noise_method == "peak2peak":
        for i, corr in enumerate(correlation):
            # compute first peak position
            (peak1_i, peak1_j), corr_max1[i] = find_first_peak(corr)

            condition = (
                corr_max1[i] < 1e-3
                or peak1_i == 0
                or peak1_i == corr.shape[0] - 1
                or peak1_j == 0
                or peak1_j == corr.shape[1] - 1
            )

            if condition:
                # return zero, since we have no signal.
                # no point to get the second peak, save time
                sig2noise[i] = 0.0
            else:
                # find second peak height
                (peak2_i, peak2_j), corr_max2 = find_second_peak(
                    corr, peak1_i, peak1_j, width=width
                )

                condition = (
                    corr_max2 == 0
                    or peak2_i == 0
                    or peak2_i == corr.shape[0] - 1
                    or peak2_j == 0
                    or peak2_j == corr.shape[1] - 1
                )
                if condition:  # mark failed peak2
                    corr_max2 = np.nan

                sig2noise[i] = corr_max1[i] / corr_max2

    elif sig2noise_method == "peak2mean":  # only one loop
        for i, corr in enumerate(correlation):
            # compute first peak position
            (peak1_i, peak1_j), corr_max1[i] = find_first_peak(corr)

            condition = (
                corr_max1[i] < 1e-3
                or peak1_i == 0
                or peak1_i == corr.shape[0] - 1
                or peak1_j == 0
                or peak1_j == corr.shape[1] - 1
            )

            if condition:
                # return zero, since we have no signal.
                # no point to get the second peak, save time
                corr_max1[i] = 0.0

        # find means of all the correlation maps
        corr_max2 = np.abs(correlation.mean(axis=(-2, -1)))
        corr_max2[corr_max2 == 0] = np.nan  # mark failed ones

        sig2noise = corr_max1 / corr_max2

    else:
        raise ValueError("wrong sig2noise_method")

    # sig2noise is zero for all failed ones
    sig2noise[np.isnan(sig2noise)] = 0.0

    return sig2noise


def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size
    Returns
    -------
        (i,j) : integers, index of the peak position
        peak  : amplitude of the peak
    """

    return np.unravel_index(np.argmax(corr), corr.shape), corr.max()


def find_second_peak(corr, i=None, j=None, width=2):
    """
    Find the value of the second largest peak.
    The second largest peak is the height of the peak in
    the region outside a 3x3 submatrxi around the first
    correlation peak.
    Parameters
    ----------
    corr: np.ndarray
          the correlation map.
    i,j : ints
          row and column location of the first peak.
    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak.
    Returns
    -------
    i : int
        the row index of the second correlation peak.
    j : int
        the column index of the second correlation peak.
    corr_max2 : int
        the value of the second correlation peak.
    """

    if i is None or j is None:
        (i, j), tmp = find_first_peak(corr)

    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as
    # masked.
    # Before check if we are not too close to the boundaries, otherwise we
    # have negative indices
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])
    tmp[iini:ifin, jini:jfin] = ma.masked
    (i, j), corr_max2 = find_first_peak(tmp)

    return (i, j), corr_max2



def get_field_shape(image_size, search_area_size, overlap):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape
    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two
    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : three elements tuple
        the shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - np.array(search_area_size)) // (
        np.array(search_area_size) - np.array(overlap)
    ) + 1

    return field_shape


def sliding_window_array(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.
    '''
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    windows = image[win_y, win_x]

    return windows

def get_rect_coordinates(frame_a, window_size, overlap, center_on_field = False):
    '''
    Rectangular grid version of get_coordinates.
    '''
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
    _, y = get_coordinates(frame_a, window_size[0], overlap[0], center_on_field = False)
    x, _ = get_coordinates(frame_a, window_size[1], overlap[1], center_on_field = False)

    return np.meshgrid(x[0,:], y[:,0])


def get_coordinates(image_size, search_area_size, overlap, center_on_field = True):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B
    overlap: int = 0 (default is no overlap)
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system
    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)

    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[1]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[0]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    if center_on_field == True:
        x += (
            image_size[1]
            - 1
            - ((field_shape[1] - 1) * (search_area_size - overlap) +
                (search_area_size - 1))
        ) // 2
        y += (
            image_size[0] - 1
            - ((field_shape[0] - 1) * (search_area_size - overlap) +
               (search_area_size - 1))
        ) // 2

        # the origin 0,0 is at top left
        # the units are pixels

    return np.meshgrid(x, y)

def normalize_intensity(window):
    """Normalize interrogation window or strided image of many windows,
       by removing the mean intensity value per window and clipping the
       negative values to zero
    Parameters
    ----------
    window :  2d np.ndarray
        the interrogation window array
    Returns
    -------
    window :  2d np.ndarray
        the interrogation window array, with mean value equal to zero and
        intensity normalized to -1 +1 and clipped if some pixels are
        extra low/high
    """
    window = window.astype(np.float32)
    window -= window.mean(axis=(-2, -1),
                          keepdims=True, dtype=np.float32)
    tmp = window.std(axis=(-2, -1), keepdims=True)
    window = np.divide(window, tmp, out=np.zeros_like(window),
                       where=(tmp != 0))
    return np.clip(window, 0, window.max())


def fft_correlate_images(image_a, image_b,
                         correlation_method="circular",
                         normalized_correlation=True,
                         conj = np.conj,
                         rfft2 = rfft2_,
                         irfft2 = irfft2_,
                         fftshift = fftshift_):
    """ FFT based cross correlation
    of two images with multiple views of np.stride_tricks()
    The 2D FFT should be applied to the last two axes (-2,-1) and the
    zero axis is the number of the interrogation window
    This should also work out of the box for rectangular windows.
    Parameters
    ----------
    image_a : 3d np.ndarray, first dimension is the number of windows,
        and two last dimensions are interrogation windows of the first image
    image_b : similar
    correlation_method : string
        one of the three methods implemented: 'circular' or 'linear'
        [default: 'circular].
    normalized_correlation : string
        decides wetehr normalized correlation is done or not: True or False
        [default: True].

    conj : function
        function used for complex conjugate

    rfft2 : function
        function used for rfft2

    irfft2 : function
        function used for irfft2

    fftshift : function
        function used for fftshift

    """

    if normalized_correlation:
        # remove the effect of stronger laser or
        # longer exposure for frame B
        # image_a = match_histograms(image_a, image_b)

        # remove mean background, normalize to 0..1 range
        image_a = normalize_intensity(image_a)
        image_b = normalize_intensity(image_b)

    s1 = np.array(image_a.shape[-2:])
    s2 = np.array(image_b.shape[-2:])

    if correlation_method == "linear":
        # have to be normalized, mainly because of zero padding
        size = s1 + s2 - 1
        fsize = 2 ** np.ceil(np.log2(size)).astype(int)
        fslice = (slice(0, image_a.shape[0]),
                  slice((fsize[0]-s1[0])//2, (fsize[0]+s1[0])//2),
                  slice((fsize[1]-s1[1])//2, (fsize[1]+s1[1])//2))
        f2a = conj(rfft2(image_a, fsize, axes=(-2, -1)))
        f2b = rfft2(image_b, fsize, axes=(-2, -1))
        corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))[fslice]
    elif correlation_method == "circular":
        f2a = conj(rfft2(image_a))
        f2b = rfft2(image_b)
        corr = fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))
    else:
        print("method is not implemented!")

    if normalized_correlation:
        corr = corr/(s2[0]*s2[1])  # for extended search area
        corr = np.clip(corr, 0, 1)
    return corr


def find_all_first_peaks(corr):
    '''
    Find row and column indices of the first correlation peak.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size
    Returns
    -------
        index_list : integers, index of the peak position in (N,i,j)
        peaks_max  : amplitude of the peak
    '''
    ind = corr.reshape(corr.shape[0], -1).argmax(-1)
    peaks = np.array(np.unravel_index(ind, corr.shape[-2:]))
    peaks = np.vstack((peaks[0], peaks[1])).T
    index_list = [(i, v[0], v[1]) for i, v in enumerate(peaks)]
    peaks_max = np.nanmax(corr, axis = (-2, -1))
    return np.array(index_list), np.array(peaks_max)

def vectorized_correlation_to_displacements(corr,
                                            n_rows = None,
                                            n_cols = None,
                                            subpixel_method = 'gaussian',
                                            eps = 1e-7
):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size

    Parameters
    ----------
    corr : 3D nd.array
        contains output of the fft_correlate_images

    n_rows, n_cols :
        number of interrogation windows, output of the get_field_shape

    mask_width: int
        distance, in pixels, from the interrogation window in which
        correlation peaks would be flagged as invalid
    Returns
    -------
    u, v: 2D nd.array
        2d array of displacements in pixels/dt
    """
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")

    corr = corr.astype(np.float32) + eps # avoids division by zero
    peaks = find_all_first_peaks(corr)[0]
    ind, peaks_x, peaks_y = peaks[:,0], peaks[:,1], peaks[:,2]
    peaks1_i, peaks1_j = peaks_x, peaks_y

    # peak checking
    if subpixel_method in ("gaussian", "centroid", "parabolic"):
        mask_width = 1
    invalid = list(np.where(peaks1_i < mask_width)[0])
    invalid += list(np.where(peaks1_i > corr.shape[1] - mask_width - 1)[0])
    invalid += list(np.where(peaks1_j < mask_width - 0)[0])
    invalid += list(np.where(peaks1_j > corr.shape[2] - mask_width - 1)[0])
    peaks1_i[invalid] = corr.shape[1] // 2 # temp. so no errors would be produced
    peaks1_j[invalid] = corr.shape[2] // 2

    print(f"Found {len(invalid)} bad peak(s)")
    if len(invalid) == corr.shape[0]: # in case something goes horribly wrong
        return np.zeros((np.size(corr, 0), 2))*np.nan

    #points
    c = corr[ind, peaks1_i, peaks1_j]
    cl = corr[ind, peaks1_i - 1, peaks1_j]
    cr = corr[ind, peaks1_i + 1, peaks1_j]
    cd = corr[ind, peaks1_i, peaks1_j - 1]
    cu = corr[ind, peaks1_i, peaks1_j + 1]

    if subpixel_method == "centroid":
        shift_i = ((peaks1_i - 1) * cl + peaks1_i * c + (peaks1_i + 1) * cr) / (cl + c + cr)
        shift_j = ((peaks1_j - 1) * cd + peaks1_j * c + (peaks1_j + 1) * cu) / (cd + c + cu)

    elif subpixel_method == "gaussian":
        inv = list(np.where(c <= 0)[0]) # get rid of any pesky NaNs
        inv += list(np.where(cl <= 0)[0])
        inv += list(np.where(cr <= 0)[0])
        inv += list(np.where(cu <= 0)[0])
        inv += list(np.where(cd <= 0)[0])

        #cl_, cr_ = np.delete(cl, inv), np.delete(cr, inv)
        #c_ = np.delete(c, inv)
        #cu_, cd_ = np.delete(cu, inv), np.delete(cd, inv)

        nom1 = log(cl) - log(cr)
        den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
        nom2 = log(cd) - log(cu)
        den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)
        shift_i = np.divide(
            nom1, den1,
            out=np.zeros_like(nom1),
            where=(den1 != 0.0)
        )
        shift_j = np.divide(
            nom2, den2,
            out=np.zeros_like(nom2),
            where=(den2 != 0.0)
        )

        if len(inv) >= 1:
            print(f'Found {len(inv)} negative correlation indices resulting in NaNs\n'+
                   'Fallback for negative indices is a 3 point parabolic curve method')
            shift_i[inv] = (cl[inv] - cr[inv]) / (2 * cl[inv] - 4 * c[inv] + 2 * cr[inv])
            shift_j[inv] = (cd[inv] - cu[inv]) / (2 * cd[inv] - 4 * c[inv] + 2 * cu[inv])

    elif subpixel_method == "parabolic":
        shift_i = (cl - cr) / (2 * cl - 4 * c + 2 * cr)
        shift_j = (cd - cu) / (2 * cd - 4 * c + 2 * cu)

    if subpixel_method != "centroid":
        disp_vy = (peaks1_i.astype(np.float64) + shift_i) - np.floor(np.array(corr.shape[1])/2)
        disp_vx = (peaks1_j.astype(np.float64) + shift_j) - np.floor(np.array(corr.shape[2])/2)
    else:
        disp_vy = shift_i - np.floor(np.array(corr.shape[1])/2)
        disp_vx = shift_j - np.floor(np.array(corr.shape[2])/2)

    disp_vx[invalid] = peaks_x[invalid]*np.nan
    disp_vy[invalid] = peaks_y[invalid]*np.nan
    #disp[ind, :] = np.vstack((disp_vx, disp_vy)).T
    #return disp[:,0].reshape((n_rows, n_cols)), disp[:,1].reshape((n_rows, n_cols))
    if n_rows == None or n_cols == None:
        return disp_vx, disp_vy
    else:
        return disp_vx.reshape((n_rows, n_cols)), disp_vy.reshape((n_rows, n_cols))



def correlation_to_displacement(corr, n_rows, n_cols,
                                subpixel_method="gaussian"):
    """
    Correlation maps are converted to displacement for each interrogation
    window using the convention that the size of the correlation map
    is 2N -1 where N is the size of the largest interrogation window
    (in frame B) that is called search_area_size
    Inputs:
        corr : 3D nd.array
            contains output of the fft_correlate_images
        n_rows, n_cols : number of interrogation windows, output of the
            get_field_shape
    """
    # iterate through interrogation widows and search areas
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # center point of the correlation map
    default_peak_position = np.floor(np.array(corr[0, :, :].shape)/2)
    for k in range(n_rows):
        for m in range(n_cols):
            # look at studying_correlations.ipynb
            # the find_subpixel_peak_position returns
            peak = np.array(find_subpixel_peak_position(corr[k*n_cols+m, :, :],
                            subpixel_method=subpixel_method)) -\
                            default_peak_position

        # the horizontal shift from left to right is the u
        # the vertical displacement from top to bottom (increasing row) is v
        # x the vertical shift from top to bottom is row-wise shift is now
        # a negative vertical
            u[k, m], v[k, m] = peak[1], peak[0]

    return (u, v)


def find_subpixel_peak_position(corr, subpixel_method="gaussian"):
    """
    Find subpixel approximation of the correlation peak.
    This function returns a subpixels approximation of the correlation
    peak by using one of the several methods available. If requested,
    the function also returns the signal to noise ratio level evaluated
    from the correlation map.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map.
    subpixel_method : string
         one of the following methods to estimate subpixel location of the
         peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.
    Returns
    -------
    subp_peak_position : two elements tuple
        the fractional row and column indices for the sub-pixel
        approximation of the correlation peak.
        If the first peak is on the border of the correlation map
        or any other problem, the returned result is a tuple of NaNs.
    """

    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.),
    # np.floor(corr.shape[1] / 2.))
    # default_peak_position = np.array([0,0])
    eps = 1e-7
    # subp_peak_position = tuple(np.floor(np.array(corr.shape)/2))
    subp_peak_position = (np.nan, np.nan)  # any wrong position will mark nan

    # check inputs
    if subpixel_method not in ("gaussian", "centroid", "parabolic"):
        raise ValueError(f"Method not implemented {subpixel_method}")

    # the peak locations
    (peak1_i, peak1_j), _ = find_first_peak(corr)

    # import pdb; pdb.set_trace()

    # the peak and its neighbours: left, right, down, up
    # but we have to make sure that peak is not at the border
    # @ErichZimmer noticed this bug for the small windows

    if ((peak1_i == 0) | (peak1_i == corr.shape[0]-1) |
       (peak1_j == 0) | (peak1_j == corr.shape[1]-1)):
        return subp_peak_position
    else:
        corr += eps  # prevents log(0) = nan if "gaussian" is used (notebook)
        c = corr[peak1_i, peak1_j]
        cl = corr[peak1_i - 1, peak1_j]
        cr = corr[peak1_i + 1, peak1_j]
        cd = corr[peak1_i, peak1_j - 1]
        cu = corr[peak1_i, peak1_j + 1]

        # gaussian fit
        if np.logical_and(np.any(np.array([c, cl, cr, cd, cu]) < 0),
                          subpixel_method == "gaussian"):
            subpixel_method = "parabolic"

        # try:
        if subpixel_method == "centroid":
            subp_peak_position = (
                ((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) /
                (cl + c + cr),
                ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) /
                (cd + c + cu),
            )

        elif subpixel_method == "gaussian":
            nom1 = log(cl) - log(cr)
            den1 = 2 * log(cl) - 4 * log(c) + 2 * log(cr)
            nom2 = log(cd) - log(cu)
            den2 = 2 * log(cd) - 4 * log(c) + 2 * log(cu)

            subp_peak_position = (
                peak1_i + np.divide(nom1, den1, out=np.zeros(1),
                                    where=(den1 != 0.0))[0],
                peak1_j + np.divide(nom2, den2, out=np.zeros(1),
                                    where=(den2 != 0.0))[0],
            )

        elif subpixel_method == "parabolic":
            subp_peak_position = (
                peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu),
            )

        return subp_peak_position



def vectorized_sig2noise_ratio(correlation,
                               sig2noise_method = 'peak2peak',
                               width = 2):
    '''
    Computes the signal to noise ratio from the correlation map in a
    mostly vectorized approach, thus much faster.
    The signal to noise ratio is computed from the correlation map with
    one of two available method. It is a measure of the quality of the
    matching between to interrogation windows.
    Parameters
    ----------
    corr : 3d np.ndarray
        the correlation maps of the image pair, concatenated along 0th axis
    sig2noise_method: string
        the method for evaluating the signal to noise ratio value from
        the correlation map. Can be `peak2peak`, `peak2mean` or None
        if no evaluation should be made.
    width : int, optional
        the half size of the region around the first
        correlation peak to ignore for finding the second
        peak. [default: 2]. Only used if sig2noise_method==peak2peak.
    Returns
    -------
    sig2noise : np.array
        the signal to noise ratios from the correlation maps.
    '''
    if sig2noise_method == "peak2peak":
        ind1, peaks1 = find_all_first_peaks(correlation)
        ind2, peaks2 = find_all_second_peaks(correlation, width = width)
        peaks1_i, peaks1_j = ind1[:, 1], ind1[:, 2]
        peaks2_i, peaks2_j = ind2[:, 1], ind2[:, 2]
        # peak checking
        flag = np.zeros(peaks1.shape).astype(bool)
        flag[peaks1 < 1e-3] = True
        flag[peaks1_i == 0] = True
        flag[peaks1_i == correlation.shape[1]-1] = True
        flag[peaks1_j == 0] = True
        flag[peaks1_j == correlation.shape[2]-1] = True
        flag[peaks2 < 1e-3] = True
        flag[peaks2_i == 0] = True
        flag[peaks2_i == correlation.shape[1]-1] = True
        flag[peaks2_j == 0] = True
        flag[peaks2_j == correlation.shape[2]-1] = True
        # peak-to-peak calculation
        peak2peak = np.divide(
            peaks1, peaks2,
            out=np.zeros_like(peaks1),
            where=(peaks2 > 0.0)
        )
        peak2peak[flag==True] = 0 # replace invalid values
        return peak2peak

    elif sig2noise_method == "peak2mean":
        peaks, peaks1max = find_all_first_peaks(correlation)
        peaks = np.array(peaks)
        peaks1_i, peaks1_j = peaks[:,1], peaks[:, 2]
        peaks2mean = np.abs(np.nanmean(correlation, axis = (-2, -1)))
        # peak checking
        flag = np.zeros(peaks1max.shape).astype(bool)
        flag[peaks1max < 1e-3] = True
        flag[peaks1_i == 0] = True
        flag[peaks1_i == correlation.shape[1]-1] = True
        flag[peaks1_j == 0] = True
        flag[peaks1_j == correlation.shape[2]-1] = True
        # peak-to-mean calculation
        peak2mean = np.divide(
            peaks1max, peaks2mean,
            out=np.zeros_like(peaks1max),
            where=(peaks2mean > 0.0)
        )
        peak2mean[flag == True] = 0 # replace invalid values
        return peak2mean
    else:
        raise ValueError(f"sig2noise_method not supported: {sig2noise_method}")


def find_all_second_peaks(corr, width = 2):
    '''
    Find row and column indices of the first correlation peak.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map fof the strided images (N,K,M) where
        N is the number of windows, KxM is the interrogation window size

    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak

    Returns
    -------
        index_list : integers, index of the peak position in (N,i,j)
        peaks_max  : amplitude of the peak
    '''
    indexes = find_all_first_peaks(corr)[0].astype(int)
    ind = indexes[:, 0]
    x = indexes[:, 1]
    y = indexes[:, 2]
    iini = x - width
    ifin = x + width + 1
    jini = y - width
    jfin = y + width + 1
    iini[iini < 0] = 0 # border checking
    ifin[ifin > corr.shape[1]] = corr.shape[1]
    jini[jini < 0] = 0
    jfin[jfin > corr.shape[2]] = corr.shape[2]
    # create a masked view of the corr
    tmp = corr.view(np.ma.MaskedArray)
    for i in ind:
        tmp[i, iini[i]:ifin[i], jini[i]:jfin[i]] = np.ma.masked
    indexes, peaks = find_all_first_peaks(tmp)
    return indexes, peaks