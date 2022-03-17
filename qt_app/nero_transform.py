# the script contains all the transformations applied within nero_app
import math
import torch
import torchvision
import numpy as np
from PIL import Image


# rotate mnist image
def rotate_mnist_image(image, angle):

    img_size = len(image)
    # rearrange image format from (28, 28, 1) to (1, 28, 28)
    image = torch.permute(image, (2, 0, 1))
    # transform includes upsample, rotate, downsample and padding (right and bottom) to image_size
    image = torchvision.transforms.Resize(img_size*3)(image)
    image = torchvision.transforms.RandomRotation(degrees=(angle, angle), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, expand=False)(image)
    image = torchvision.transforms.Resize(img_size)(image)
    # permute back
    image = torch.permute(image, (1, 2, 0))

    return image

# prepare mnist image tensor
def prepare_mnist_image(image):
    # rearrange image format from (28, 28, 1) to (1, 28, 28)
    image = torch.permute(image, (2, 0, 1)).float()
    image = torchvision.transforms.Normalize((0.1307,), (0.3081,))(image)
    image = torchvision.transforms.Pad((0, 0, 1, 1), fill=0, padding_mode='constant')(image)
    # permute back
    image = torch.permute(image, (1, 2, 0))

    return image


# load the class names from the class file
def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


# perform jittering with a specific x_tran and y_tran during the aggregate test phase
class FixedJittering(object):
    def __init__(self, img_size, x_tran, y_tran):

        self.img_size = img_size
        self.x_tran = x_tran
        self.y_tran = y_tran

    # helper function that computes labels for cut out images
    def compute_label(self, cur_bounding_box, x_min, y_min, image_size):
        # convert key object bounding box to be based on extracted image
        bb_x_min = cur_bounding_box[0] - x_min
        bb_y_min = cur_bounding_box[1] - y_min
        bb_x_max = cur_bounding_box[2] - x_min
        bb_y_max = cur_bounding_box[3] - y_min

        # compute the center of the object in the extracted image
        object_center_x = (bb_x_min + bb_x_max) / 2
        object_center_y = (bb_y_min + bb_y_max) / 2

        # compute the width and height of the real bounding box of this object
        original_bb_width = cur_bounding_box[2] - cur_bounding_box[0]
        original_bb_height = cur_bounding_box[3] - cur_bounding_box[1]

        # compute the range of the bounding box, do the clamping if go out of extracted image
        bb_min_x = max(0, object_center_x - original_bb_width/2)
        bb_max_x = min(image_size-1, object_center_x + original_bb_width/2)
        bb_min_y = max(0, object_center_y - original_bb_height/2)
        bb_max_y = min(image_size-1, object_center_y + original_bb_height/2)

        return bb_min_x, bb_min_y, bb_max_x, bb_max_y

    # helper function that extracts the FOV from original image, padding if needed
    def extract_img_with_pad(self, original_img, extracted_img_size, x_min, y_min, x_max, y_max):
        x_min = round(x_min)
        y_min = round(y_min)
        x_max = round(x_max)
        y_max = round(y_max)
        height, width = original_img.shape[:2]
        extracted_img = np.zeros((extracted_img_size, extracted_img_size, 3), dtype=np.uint8)

        # completed within
        if x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            extracted_img = original_img[y_min:y_max, x_min:x_max, :]

        # if top left part went out
        elif x_min < 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top left went out')
            extracted_img[abs(y_min):, abs(x_min):, :] = original_img[0:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:abs(y_min), :abs(x_min), :] = original_img[0:abs(y_min), 0:abs(x_min), :]
            extracted_img[abs(y_min):, :abs(x_min), :] = original_img[0:y_max, 0:abs(x_min), :]
            extracted_img[:abs(y_min), abs(x_min):, :] = original_img[0:abs(y_min), 0:x_max, :]

        # if left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max < height-1:
            # print('left went out')
            extracted_img[:, abs(x_min):, :] = original_img[y_min:y_max, 0:x_max, :]
            # mirror padding
            extracted_img[:, :abs(x_min), :] = original_img[y_min:y_max, 0:abs(x_min), :]

        # if bottom left part went out
        elif x_min < 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom left went out')
            extracted_img[0:extracted_img_size-(y_max-height), abs(x_min):, :] = original_img[y_min:, 0:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, 0:abs(x_min), :] = original_img[2*height-y_max:, 0:abs(x_min), :]
            extracted_img[0:extracted_img_size-(y_max-height), 0:abs(x_min), :] = original_img[y_min:, 0:abs(x_min), :]
            extracted_img[extracted_img_size-(y_max-height):, abs(x_min):, :] = original_img[2*height-y_max:, 0:extracted_img_size-abs(x_min), :]

        # if bottom part went out
        elif x_min >= 0 and y_min >= 0 and x_max < width-1 and y_max > height-1:
            # print('bottom went out')
            extracted_img[0:extracted_img_size-(y_max-height), :, :] = original_img[y_min:, x_min:x_max, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, :, :] = original_img[2*height-y_max:, x_min:x_max, :]

        # if bottom right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max > height-1:
            # print('bottom right went out')
            extracted_img[0:extracted_img_size-(y_max-height), 0:extracted_img_size-(x_max-width), :] = original_img[y_min:, x_min:, :]
            # mirror padding
            extracted_img[extracted_img_size-(y_max-height):, extracted_img_size-(x_max-width):, :] = original_img[height-y_max:, 2*width-x_max:, :]
            extracted_img[extracted_img_size-(y_max-height):, 0:extracted_img_size-(x_max-width), :] = original_img[height-y_max:, x_min:, :]
            extracted_img[0:extracted_img_size-(y_max-height), extracted_img_size-(x_max-width):, :] = original_img[y_min:, 2*width-x_max:, :]

        # if right part went out
        elif x_min >= 0 and y_min >= 0 and x_max > width-1 and y_max < height-1:
            # print('\nright went out')
            extracted_img[:, 0:extracted_img_size-(x_max-width), :] = original_img[y_min:y_max, x_min:, :]
            # mirror padding
            extracted_img[:, extracted_img_size-(x_max-width):, :] = original_img[y_min:y_max, 2*width-x_max:, :]

        # if top right part went out
        elif x_min >= 0 and y_min < 0 and x_max >= width-1 and y_max < height-1:
            # print('top right went out')
            extracted_img[abs(y_min):, 0:extracted_img_size-(x_max-width), :] = original_img[0:y_max, x_min:, :]
            # mirror padding
            extracted_img[0:abs(y_min), extracted_img_size-(x_max-width):, :] = original_img[0:abs(y_min), 2*width-x_max:, :]
            extracted_img[0:abs(y_min), 0:extracted_img_size-(x_max-width), :] = original_img[0:abs(y_min), x_min:, :]
            extracted_img[abs(y_min):, extracted_img_size-(x_max-width):, :] = original_img[0:y_max, 2*width-x_max:, :]

        # if top part went out
        elif x_min >= 0 and y_min < 0 and x_max < width-1 and y_max < height-1:
            # print('top went out')
            extracted_img[abs(y_min):, :, :] = original_img[0:y_max, x_min:x_max, :]
            # mirror padding
            extracted_img[0:abs(y_min), :, :] = original_img[0:abs(y_min), x_min:x_max, :]

        return extracted_img


    def __call__(self, data):
        label_path, img, labels = data

        # all_ids = labels[:, 0].astype(int)
        # all_bbs = labels[:, 1:]
        # # analyze the img path to get the key bb index
        # key_label_index = int(label_path.split('_')[-1].split('.')[0])
        # # get the target bb
        # key_id = int(all_ids[key_label_index])
        # key_bb = all_bbs[key_label_index]


        # all_bbs only contains 1 bounding box
        key_id = labels[0, -1].astype(int)
        key_bb = labels[0, :4]

        # labels for current image
        processed_labels = []

        # center of the target object
        key_bb_min_x = key_bb[0]
        key_bb_min_y = key_bb[1]
        key_bb_max_x = key_bb[2]
        key_bb_max_y = key_bb[3]
        key_center_x = (key_bb_min_x + key_bb_max_x) / 2
        key_center_y = (key_bb_min_y + key_bb_max_y) / 2

        # the cut-out image positions in the original image
        # the move of actual object and the window are opposite, thus minus
        x_min = key_center_x - self.img_size//2 - self.x_tran
        y_min = key_center_y - self.img_size//2 - self.y_tran
        x_max = key_center_x + self.img_size//2 - self.x_tran
        y_max = key_center_y + self.img_size//2 - self.y_tran

        # compute the bounding box wrt extrated image
        key_bb_min_x, key_bb_min_y, key_bb_max_x, key_bb_max_y \
            = self.compute_label(key_bb, x_min, y_min, self.img_size)

        # put label in the format of (x_1, y_1, x_2, y_2, class)
        processed_labels.append([key_bb_min_x,
                                key_bb_min_y,
                                key_bb_max_x,
                                key_bb_max_y,
                                key_id])

        # extract the image, pad if needed
        extracted_img = self.extract_img_with_pad(img, self.img_size, x_min, y_min, x_max, y_max)
        processed_labels = np.array(processed_labels)

        return label_path, extracted_img, processed_labels


# convert object classes from coco to custom class labels
class ConvertLabel(object):
    def __init__(self, original_classes, desired_classes):

        self.original_classes = original_classes
        self.desired_classes = desired_classes

    def __call__(self, data):

        # for this study, each image has one label only
        label_path, img, labels = data

        # convert key id from COCO classes to custom desired classes
        for i, cur_label in enumerate(labels):
            # 0 is always background for pytorch faster-rcnn
            cur_id = cur_label[-1].astype(int)
            key_id_custom = int(self.desired_classes.index(self.original_classes[cur_id]) + 1)
            labels[i, -1] = key_id_custom

        return label_path, img, labels


# convert both images and bounding boxes to PyTorch tensors (used in aggregate case)
class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        label_path, img, boxes = data
        # Extract image as PyTorch tensor
        img = torchvision.transforms.ToTensor()(img)

        # add index in front of the label tensor
        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = torchvision.transforms.ToTensor()(boxes)

        return img, bb_targets


# rotates the PIV images and ground truths in terms of time (velocity only)
def rotate_piv_data(all_images, all_labels, degree, sanity_check=False, vis=False):

    # single input images case
    if len(all_images.shape) == 3:
        rotated_image = torch.rot90(all_images, int(degree//90))
        # rotate the label (counterclock wise)
        label_rotated_temp = torch.rot90(all_labels, int(degree//90), axes=(0, 1))
        # rotate each velocity vector too (counterclock wise)
        label_rotated = torch.zeros(all_labels.shape)
        label_rotated[:, :, 0] = label_rotated_temp[:, :, 0] * np.cos(math.radians(degree)) + label_rotated_temp[:, :, 1] * np.sin(math.radians(degree))
        label_rotated[:, :, 1] = -label_rotated_temp[:, :, 0] * np.sin(math.radians(degree)) + label_rotated_temp[:, :, 1] * np.cos(math.radians(degree))

        return rotated_image, label_rotated

    # multiple input images case
    elif len(all_images.shape) == 4:
        # check if lengths match
        num_images = len(all_images)
        num_labels = len(all_labels)
        if (num_images != num_labels):
            raise Exception(f'generate_pt_dataset: Images size {num_images} does not \
                                match labels size {num_labels}')
        else:
            print(f'Number of images and labels is {num_images}')


        rotated_images = torch.zeros(all_images.shape)
        rotated_labels = torch.zeros(all_labels.shape)


        # rotate the image and label
        for i in range(num_images):

            # about image
            cur_image = all_images[i]
            # rotate the image (only support 90, 180, 270, etc) (counterclock wise)
            rotated_images[i] = torch.rot90(cur_image, int(degree//90))

            # about labels
            cur_label = all_labels[i]
            # rotate the label (counterclock wise)
            cur_label_rotated_temp = torch.rot90(cur_label, int(degree//90), axes=(0, 1))
            # rotate each velocity vector too (counterclock wise)
            cur_label_rotated = torch.zeros(cur_label.shape)
            cur_label_rotated[:, :, 0] = cur_label_rotated_temp[:, :, 0] * np.cos(math.radians(degree)) + cur_label_rotated_temp[:, :, 1] * np.sin(math.radians(degree))
            cur_label_rotated[:, :, 1] = -cur_label_rotated_temp[:, :, 0] * np.sin(math.radians(degree)) + cur_label_rotated_temp[:, :, 1] * np.cos(math.radians(degree))
            rotated_labels[i] = cur_label_rotated


        return rotated_images, rotated_labels
