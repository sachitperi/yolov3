
1) darknet.py
    * Name of the underlying architecture of yolo. 
    File contains the code that will create yolo architecture
    * contains the following functions
        * parse_cfg: Parse the config file and store the every block as a dict. 
        The attributes of blocks and their values are stored as key value pairs in the dict.
        As we parse through the config we keep appending these dicts denoted by the variable `block` in the `code`, to a list of blocks.
    
2) util.py  
    * to suppliment the darknet.py file.
    File will contains code for various helper functions
    
3) cfg directory
    * Official code (authored in C) uses a configuration file to build the network
    The file is equivalent to .protxt used to describe the networks.
    * The link to the file can be found in pjreddie's github account
        * `https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg` 
        * `wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg`
        
    * The cfg file contains information about the network various layers used in the network
        * There are 5 types of layers that are used in yolo
            * Convolutional:
                example
                ```
                    [convolutional]
                    batch_normalize=1
                    filters=32
                    size=3
                    stride=1
                    pad=1
                    activation=leaky
                ```
            * Shortcut: A shortcut layer is a skip connection like the one used in ResNet.
            The from parameter -3 indicates that the output of shortcut layer is obtained 
            by adding previous and 3rd layer backwards from the shortcut layer
                example
                ```
                    [shortcut]
                    from=-3
                    activation=linear
                ```
            * Upsample: The layer upsamples the featuremap from previous layer by a 
            factor of stride using bi-linear upsampling
            ```
                [upsample]
                stride=2
            ```
            * Route: The route layer has an attribute layers which can have either 1 0r 2 values.  
                * When `layers` attribute has 1 value it outputs featuremaps of the layer indexed by the value.  
                In the below example the value is `-4` which means the layer will output featuremap from 4th layer backwards from route layer. 
                * When `layers` attribute has two values, it returns the concatenated feature maps of the layers indexed by its values.  
                In the below example it is -1 and 61 hence route layer will output the featuremaps from previous layer (-1) and 61st layer, concatenated along with depth dimension.  
            ```
                [route]
                layers = -4
                
                
                [route]
                layers = -1, 61
            ```
            * Yolo: yolo layer corresponds to the detection layer.  
            The `anchors` describes 9 anchors, but only anchors which are indexed by attributes of the `mask` tag are used.  
            In the below example `mask` is 3, 4, 5 which means the fourth, fifth and sixth anchors are used. This makes sense since each detection layer predicts 3 bounding boxes.
            In total we have detection layers at 3 scales, making up for 9 anchors.
            ```
                [yolo]
                mask = 3,4,5
                anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                classes=80
                num=9
                jitter=.3
                ignore_thresh = .7
                truth_thresh = 1
                random=1
            ```
            * Net: The net block is not a layer but only describes information about the network input and training parameters.  
            It is not used in forward pass of the yolo but it does provide us information like the network inputsize, which we use to adjust anchors in forward pass.  
            ```python
                [net]
                # Testing
                # batch=1
                # subdivisions=1
                # Training
                batch=64
                subdivisions=16
                width=608
                height=608
                channels=3
                momentum=0.9
                decay=0.0005
                angle=0
                saturation = 1.5
                exposure = 1.5
                hue=.1
            ```                          
4)  Defining the Network 