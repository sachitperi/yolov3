from __future__ import  division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy as np

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

blocks = parsecfg("./cfg/yolov3.cfg")
# print(blocks)
print(create_modules(blocks))