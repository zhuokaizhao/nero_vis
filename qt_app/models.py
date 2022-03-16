import torch
import e2cnn
import numpy as np
import torchvision
from e2cnn import gspaces
import torch.utils.model_zoo as model_zoo
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import correlation
import layers


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
# PIV-LiteFlowNet-En (Cai)
class PIV_LiteFlowNet_en(torch.nn.Module):
    def __init__(self, **kwargs):
        # use cudnn
        torch.backends.cudnn.enabled = True
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