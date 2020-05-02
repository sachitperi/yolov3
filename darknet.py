from __future__ import  division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
import cv2

def parsecfg(cfgfile):
    """
    :param cfgfile: Path to the configuration file
    :return: List of blocks. Each blocks describes a block in the neural network to be built.
    Block is represented as a dictionary in the list
    """
    file = open(cfgfile, 'r')
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]

    # loop over the resultant list to get blocks
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block)!=0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value
    blocks.append(block)
    return blocks

# create pytorch modules for blocks present in cfg file using output from parsecfg()
def create_modules(blocks):
    """
    :param blocks:
    :return: nn.Module_List()

    net_info: used to store information about the network
    prev_filters: Height and width of a conv kernel is defined in the cfg file. but depth is not hence the dept id number of feature maps in the previous layer. This is stored in this variable.
        Also by default it is initilized to 3 because it has 3 filters. And Route layer brings (possibly concatenated) feature maps from previous layers.
        If there is a conv layer right in front of the route layer, then kernel is applied on the feature maps of previous layers,
        precisely the ones the route layer begins with. Hence need to keep track of the number of filters in not only the previous layer,
        but also in each one of the preceding layer
    output_filters: As we iterate we append the number of output filters of each block to the list

    nn.Sequential class used to sequentially execute a number of nn.Module objects
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        # check the type of the block
        # create a new module for the block
        # append to module_list

        if (x["type"]=="convolutional"):
            # get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the batchnorm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # check the activation
            # it is either Linear or leaky Relu for yolo
            if activation=="leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # if its an upsampling layer
        # we use Binary2dUpsampling
        elif(x["type"]=="upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # if it is a route layer
        elif(x["type"]=="route"):
            x["layers"] = x["layers"].split(",")
            # start of a route
            start = int(x["layers"][0])
            # end if there exists one
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut corresponds to skip connections
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        # Yolo is the detection layer
        elif x["type"]=="yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        # bookkeeping at the end of the loop
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

# DetectionLayer that holds anchors used to detect bounding boxes
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parsecfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    # implimenting the forward pass of the network
    def forward(self, x, CUDA):
        """
        The forward pass serves two purpose first to calculate the output and second to transform the output detection featuremaps in a way that it can be processed
        easier (eg: transforming them such that detection across multiple scales can be concatenated, which otherwise isnt possible as they are of different dimensions)

        Since route and shortcut layers need output maps from previous layers, we cache the output feature maps of every layer in a dict `outputs`.
        Keys are indices of the layer and values are the feature maps

        Iterate over module_list which contains the modules of the network. The modules have been appended in the same order as they are present in the config file.
        This means we can simply run our input through each module.

        :param x: input x
        :param CUDA: if true would use gpu
        :return:
        """
        modules = self.blocks[1:]
        outputs = {} # Cache the outputs for route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # if the module is a convolutional or upsample layer this is how the forward pass should work
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # route layer and shortcut layer
            # In Route layer we have to account for two cases.
            # For cases where we have to concatenate the feature maps we use torch.cat function with second argument as 1.
            # This is because we want to concatenate the feature maps along the depth. (Pytorch has input and output of convolutionals layer has the format BxCxHxW.
            # The depth corresponding to the channel dimension)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            # Yolo detection layer
            # The output of yolo is a convolutional featuremap that contains the bounding box attributes along the depth of the featuremap
            # The attributes bounding-box predicted by a cell are stacked one by one along each other
            # second bounding of a cell at (5, 6) will have to index it by map [5, 6, (5+c): 2*(5+c)]
            # This form is inconvineant for output processing such as thresholding by output confidence, adding grid ofsets to centers, applying anchors etc
            # Also since detections happen at 3 scales the dimensions of prediction maps will be different.
            # Although the dimensions of featuremaps are different the output processing operations on them are similar. It would be nice to do these operations on a single tensor,
            # rather than three separate tensors
            # for this the function predict_transforms from utils.py is used to transform the output tensor a table with bounding boxes as its rows an concatenating is easy
            # An obstacle is that we cannot initialize an empty tensor and then concatenate a non-empty tensor (of different shape) to it. so delay initialization until we get our first detection map
            # write=0 flag is used to indicate whether we have reached the first detection or not.

            elif module_type=="yolo":
                anchors=self.module_list[i][0].anchors
                # get input dimensions
                inp_dim = int(self.net_info["height"])
                # get the number of classes
                num_classes = int(module["classes"])
                # Transform
                x = x.data
                x = predict_transforms(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)
            outputs[i] = x
        return detections

    # testing forwardpass

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416)) # resize the input dimension
    img_ = img[:,:,::-1].transpose((2, 0, 1)) # BGR -> RGB | HxWXC -> CxHXW
    img_ = img_[np.newaxis, :, :, :]/255.0 # Add a channel at 0 for batch | Normalize
    img_ = torch.from_numpy(img_).float() # convert to float
    img_ = Variable(img_)
    return img_

# Main
# blocks = parsecfg("./cfg/yolov3.cfg")
# # print(blocks)
# print(create_modules(blocks))
model = Darknet("./cfg/yolov3.cfg")
inp = get_test_input()
# print(inp)
pred = model(inp, torch.cuda.is_available())
print(pred)