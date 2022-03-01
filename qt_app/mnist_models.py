import torch
import e2cnn
import numpy as np
from e2cnn import gspaces
import torch.utils.model_zoo as model_zoo


######################################## For MNIST dataset ########################################
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


