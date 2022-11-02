import torch
# from torch._C import T
import e2cnn
import torchvision
import numpy as np
from e2cnn import gspaces
import torch.utils.model_zoo as model_zoo

# For SESN
# from impl.ses_conv import SESMaxProjection
# from impl.ses_conv import SESConv_Z2_H, SESConv_H_H

# For DSS
# from impl.deep_scale_space import Dconv2d, BesselConv2d, ScaleMaxProjection

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = torch.nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = torch.nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = torch.nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

# translation equivariant downsampling layer
class Downsample(torch.nn.Module):
    def __init__(self, pad_type='zero', filt_size=3, stride=2, channels=None, pad_off=0): #reflect
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)), int(np.ceil(1.*(filt_size-1))), int(1.*(filt_size-1)), int(np.ceil(1.*(filt_size-1)))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            # input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
            return torch.nn.functional.F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


######################################## For MNIST dataset ########################################
# non-rotation nor translation equivariant network
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


# translation equivariant network using full-convolution techniques
class Shift_Eqv_Net_MNIST(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(Shift_Eqv_Net_MNIST, self).__init__()
        # convolution 1
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=24,
                            kernel_size=7,
                            stride=1,
                            padding=2,
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
            # stride-2 AvgPool is replaced by stride-1 AvgPool+Downsample
            torch.nn.AvgPool2d(kernel_size=2, stride=1),
            Downsample(filt_size=1, stride=2, channels=1)
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
            # stride-2 AvgPool is replaced by stride-1 AvgPool+Downsample
            torch.nn.AvgPool2d(kernel_size=2, stride=1),
            Downsample(filt_size=1, stride=2, channels=1)
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
                            padding=2,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True)
        )

        self.pool3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=1)
        )

        # fully connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(14400, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )


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


# non-rotation nor translation equivariant network
class Scale_Eqv_Net_MNIST(torch.nn.Module):
    def __init__(self, method, n_classes=10):
        super(Scale_Eqv_Net_MNIST, self).__init__()
        if method == 'SESN':
            num_scale = 7
            scales = [1.0]
            # convolution 1
            self.block1 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=1,
                                out_channels=24,
                                kernel_size=7,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=1,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=24),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 2
            self.block2 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=24,
                                out_channels=48,
                                kernel_size=5,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=2,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=48),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 3
            self.block3 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=48,
                                out_channels=48,
                                kernel_size=5,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=2,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=48),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 4
            self.block4 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=48,
                                out_channels=96,
                                kernel_size=5,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=2,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=96),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 5
            self.block5 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=96,
                                out_channels=96,
                                kernel_size=5,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=2,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=96),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 6
            self.block6 = torch.nn.Sequential(
                SESConv_Z2_H(in_channels=96,
                                out_channels=64,
                                kernel_size=5,
                                effective_size=num_scale,
                                scales=scales,
                                stride=1,
                                padding=1,
                                bias=False),
                SESMaxProjection(),
                torch.nn.BatchNorm2d(num_features=64),
                torch.nn.ReLU(inplace=True)
            )

        elif method == 'DSS':
            n_scales = 4
            scale_sizes = 1
            init = 'he'
            # convolution 1
            self.block1 = torch.nn.Sequential(
                BesselConv2d(n_channels=1, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(1, 24, kernel_size=[scale_sizes, 7, 7], base=2,
                        io_scales=[n_scales, n_scales], padding=1, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=24),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 2
            self.block2 = torch.nn.Sequential(
                BesselConv2d(n_channels=24, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(24, 48, kernel_size=[scale_sizes, 5, 5], base=2,
                        io_scales=[n_scales, n_scales], padding=2, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=48),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 3
            self.block3 = torch.nn.Sequential(
                BesselConv2d(n_channels=48, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(48, 48, kernel_size=[scale_sizes, 5, 5], base=2,
                        io_scales=[n_scales, n_scales], padding=2, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=48),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 4
            self.block4 = torch.nn.Sequential(
                BesselConv2d(n_channels=48, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(48, 96, kernel_size=[scale_sizes, 5, 5], base=2,
                        io_scales=[n_scales, n_scales], padding=2, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=96),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 5
            self.block5 = torch.nn.Sequential(
                BesselConv2d(n_channels=96, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(96, 96, kernel_size=[scale_sizes, 5, 5], base=2,
                        io_scales=[n_scales, n_scales], padding=2, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=96),
                torch.nn.ReLU(inplace=True)
            )

            # convolution 6
            self.block6 = torch.nn.Sequential(
                BesselConv2d(n_channels=96, base=2, zero_scale=0.25, n_scales=n_scales),
                Dconv2d(96, 64, kernel_size=[scale_sizes, 5, 5], base=2,
                        io_scales=[n_scales, n_scales], padding=1, init=init),
                ScaleMaxProjection(),
                torch.nn.BatchNorm2d(num_features=64),
                torch.nn.ReLU(inplace=True)
            )

        self.pool1 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.pool2 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.pool3 = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=1)
        )

        # fully connected
        if method == 'SESN':
            self.fully_net = torch.nn.Sequential(
                torch.nn.Linear(4096, 64),
                # torch.nn.Linear(6400, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(64, n_classes),
            )
        elif method == 'DSS':
            self.fully_net = torch.nn.Sequential(
                torch.nn.Linear(2304, 64),
                # torch.nn.Linear(6400, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ELU(inplace=True),
                torch.nn.Linear(64, n_classes),
            )


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

        return x



####################################### For CIFAR-10 Dataset #######################################
# translation equivariant VGG with full convolution
class VGG(torch.nn.Module):

    def __init__(self, features, eqv_mode, num_rotation=None, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # trans-only method which uses full-convolution does not have a influence here
        if eqv_mode == 'trans-eqv' or 'non-eqv':
            self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
            # image net classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, num_classes)
            )
        elif eqv_mode == 'rot-eqv':
            assert num_rotation != None
            # the model is equivariant under rotations by 45 degrees, modelled by C8
            r2_act = gspaces.Rot2dOnR2(N=num_rotation)
            in_type = e2cnn.nn.FieldType(r2_act, 3*[r2_act.trivial_repr])
            self.avgpool = e2cnn.nn.PointwiseAvgPool(in_type, (7, 7))
            # image net classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, 4096),
                e2cnn.nn.ReLU(in_type, True),
                e2cnn.nn.PointwiseDropout(in_type),
                torch.nn.Linear(4096, 4096),
                e2cnn.nn.ReLU(in_type, True),
                e2cnn.nn.PointwiseDropout(in_type),
                torch.nn.Linear(4096, num_classes)
            )
        else:
            raise Exception(f'Unknown eqv_mode {eqv_mode}')

        # CIFAR classifier
        # self.classifier = torch.nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


def make_layers(cfg, eqv_mode, image_size, num_groups=None, batch_norm=False, filter_size=1):
    layers = []
    in_channels = 3

    for i, v in enumerate(cfg):
        if eqv_mode == 'rot-eqv':
            r2_act = gspaces.Rot2dOnR2(N=num_groups)
            if i == 0:
                in_type = e2cnn.nn.FieldType(r2_act, 3*[r2_act.trivial_repr])
                if image_size != None:
                    layers += [e2cnn.nn.MaskModule(in_type, image_size[0], margin=image_size[0]-32)]
                else:
                    layers += [e2cnn.nn.MaskModule(in_type, 32, margin=0)]
            else:
                in_type = e2cnn.nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])

        if v == 'M':
            if eqv_mode == 'non-eqv':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            elif eqv_mode == 'trans-eqv':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=1), Downsample(filt_size=filter_size, stride=2, channels=in_channels)]
            elif eqv_mode == 'rot-eqv':
                layers += [e2cnn.nn.PointwiseMaxPool(in_type, kernel_size=2, stride=2)]
            else:
                raise Exception(f'Unrecognized eqv_mode {eqv_mode}')

        else:
            # eqv model essentially make padding = 2, non-eqv model has padding = 1
            if eqv_mode == 'non-eqv':
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            elif eqv_mode == 'trans-eqv':
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=2)
            elif eqv_mode == 'rot-eqv':
                out_type = e2cnn.nn.FieldType(r2_act, v*[r2_act.regular_repr])
                conv2d = e2cnn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=2)

            if batch_norm:
                if eqv_mode == 'rot-eqv':
                    layers += [conv2d, e2cnn.nn.InnerBatchNorm(out_type), e2cnn.nn.ReLU(in_type, True)]
                else:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                if eqv_mode == 'rot-eqv':
                    layers += [conv2d, e2cnn.nn.ReLU(in_type, True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]

            in_channels = v

    return torch.nn.Sequential(*layers)


# coefficients for different variants of VGG
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def vgg11_bn(eqv_mode, image_size, num_groups=None, pretrained=False, filter_size=1, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], eqv_mode, image_size, num_groups=num_groups, batch_norm=True, filter_size=filter_size), eqv_mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model

def vgg19_bn(eqv_mode, image_size, num_groups=None, pretrained=False, filter_size=1, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], eqv_mode, image_size, num_groups=num_groups, batch_norm=True, filter_size=filter_size), eqv_mode, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

