# the script generates faster rcnn models with input backbone name
import torch
import torchvision
from  torchvision.models.detection import backbone_utils
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# faster rcnn with mobilenetv2 backbone
def faster_rcnn(num_classes, backbone_name, img_size, pretrained_backbone=True):

    # load a pre-trained model for classification and return only the features
    if backbone_name == 'mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained_backbone).features
        # FasterRCNN needs to know the number of output channels in a backbone.
        # For mobilenet_v2, it's 1280, so we need to add it here
        backbone.out_channels = 1280
        # let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
        # We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes andaspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        # define what are the feature maps that we will use to perform the region of interest cropping,
        # as well as the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to be [0].
        # More generally, the backbone should return an OrderedDict[Tensor],
        # and in featmap_names you can choose which feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
    elif 'resnet' in backbone_name:
        backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained_backbone, trainable_layers=3)

        anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                            aspect_ratios=tuple([(0.5, 1.0, 2.0) for _ in range(5)]))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                        output_size=7,
                                                        sampling_ratio=2)



    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                        num_classes=num_classes+1,
                        rpn_anchor_generator=anchor_generator,
                        box_roi_pool=roi_pooler,
                        min_size=img_size,
                        max_size=img_size)

    print(f'Faster-RCNN with {backbone_name} backbone is generated')

    return model

