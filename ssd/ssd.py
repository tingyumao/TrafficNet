import tensorflow as tf
from ssd.ssd_layers import Normalize
from ssd.ssd_layers import PriorBox
from ssd.ssd_layers import AtrousConvolution2D

class SSD(object):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        self.fc6 = AtrousConvolution2D((3,3,512,1024), name="fc6")
        self.conv4_3_norm = Normalize(20, name="conv4_3_norm")
        
    def __call__(self, input_tensor, PRIORS=True):
        
        input_shape = self.input_shape
        num_classes = self.num_classes
        
        # input image size
        img_size = (input_shape[1], input_shape[0])
        # block1
        h_conv1_1 = tf.layers.conv2d(input_tensor, 64, [3, 3], padding="same",name="conv1_1", activation=tf.nn.relu)
        h_conv1_2 = tf.layers.conv2d(h_conv1_1, 64, [3, 3], padding="same",name="conv1_2", activation=tf.nn.relu)
        h_pool1 = tf.layers.max_pooling2d(h_conv1_2, 2, 2, name="pool1", padding="same")
        
        # block2
        h_conv2_1 = tf.layers.conv2d(h_pool1, 128, [3, 3], padding="same",name="conv2_1", activation=tf.nn.relu)
        h_conv2_2 = tf.layers.conv2d(h_conv2_1, 128, [3, 3], padding="same",name="conv2_2", activation=tf.nn.relu)
        h_pool2 = tf.layers.max_pooling2d(h_conv2_2, 2, 2, name="pool2", padding="same")
        
        # block3
        h_conv3_1 = tf.layers.conv2d(h_pool2, 256, [3, 3], padding="same",name="conv3_1", activation=tf.nn.relu)
        h_conv3_2 = tf.layers.conv2d(h_conv3_1, 256, [3, 3], padding="same",name="conv3_2", activation=tf.nn.relu)
        h_conv3_3 = tf.layers.conv2d(h_conv3_2, 256, [3, 3], padding="same",name="conv3_3", activation=tf.nn.relu)
        h_pool3 = tf.layers.max_pooling2d(h_conv3_3, 2, 2, name="pool3", padding="same")
        
        # block4
        h_conv4_1 = tf.layers.conv2d(h_pool3, 512, [3, 3], padding="same",name="conv4_1", activation=tf.nn.relu)
        h_conv4_2 = tf.layers.conv2d(h_conv4_1, 512, [3, 3], padding="same",name="conv4_2", activation=tf.nn.relu)
        h_conv4_3 = tf.layers.conv2d(h_conv4_2, 512, [3, 3], padding="same",name="conv4_3", activation=tf.nn.relu)
        h_pool4 = tf.layers.max_pooling2d(h_conv4_3, 2, 2, name="pool4", padding="same")
        
        # block5
        h_conv5_1 = tf.layers.conv2d(h_pool4, 512, [3, 3], padding="same",name="conv5_1", activation=tf.nn.relu)
        h_conv5_2 = tf.layers.conv2d(h_conv5_1, 512, [3, 3], padding="same",name="conv5_2", activation=tf.nn.relu)
        h_conv5_3 = tf.layers.conv2d(h_conv5_2, 512, [3, 3], padding="same",name="conv5_3", activation=tf.nn.relu)
        h_pool5 = tf.layers.max_pooling2d(h_conv5_3, 3, 1, name="pool5", padding="same")
        
        # FC6 (Full-Context?)
        #fc6 = AtrousConvolution2D((3,3,512,1024), name="fc6")
        fc6 = self.fc6
        h_fc6 = fc6(h_pool5)
        
        # FC7
        h_fc7 = tf.layers.conv2d(h_fc6, 1024, [1, 1], padding="same",name="fc7", activation=tf.nn.relu)
        
        # block6, subsampling
        h_conv6_1 = tf.layers.conv2d(h_fc7, 256, [1, 1], padding="same",name="conv6_1", activation=tf.nn.relu)
        h_conv6_2 = tf.layers.conv2d(h_conv6_1, 512, [3, 3], strides=(2, 2), padding="same",name="conv6_2", activation=tf.nn.relu)
        
        # block7, subsampling
        h_conv7_1 = tf.layers.conv2d(h_conv6_2, 128, [1, 1], padding="same",name="conv7_1", activation=tf.nn.relu)
        h_conv7_1 = tf.keras.layers.ZeroPadding2D()(h_conv7_1)
        h_conv7_2 = tf.layers.conv2d(h_conv7_1, 256, [3, 3], strides=(2, 2), padding="valid",name="conv7_2", activation=tf.nn.relu) # maybe "same" is also fine?
        
        # block8
        h_conv8_1 = tf.layers.conv2d(h_conv7_2, 128, [1, 1], padding="same",name="conv8_1", activation=tf.nn.relu)
        h_conv8_2 = tf.layers.conv2d(h_conv8_1, 256, [3, 3], strides=(2, 2), padding="same",name="conv8_2", activation=tf.nn.relu)
        
        # last pool(pool6), global average pool2d, return (1,1,channels)
        #print(tf.shape(h_conv8_2))
        #height, width = tf.to_int32(tf.shape(h_conv8_2)[1]), tf.to_int32(tf.shape(h_conv8_2)[2])
        #print(height)
        #print(width)
        #_, height, width, _ = tf.shape(h_conv8_2)
        #h_pool6 = tf.layers.average_pooling2d(h_conv8_2, [height, width], [height,width])
        h_pool6 = tf.reduce_mean(h_conv8_2, [1,2])
        
        ###########################
        # prediction from conv4_3##
        ###########################
        h_conv4_3_norm = self.conv4_3_norm(h_conv4_3)
        num_priors = 3
        ## loc
        h_conv4_3_norm_mbox_loc = tf.layers.conv2d(h_conv4_3_norm, num_priors*4, [3, 3], padding="same",name="conv4_3_norm_mbox_loc", activation=None)
        ## flatten: y = tf.layers.Flatten()(x)
        h_conv4_3_norm_mbox_loc_flat = tf.layers.Flatten()(h_conv4_3_norm_mbox_loc)
        ## conf
        name = "conv4_3_norm_mbox_conf_{}".format(num_classes)
        h_conv4_3_norm_mbox_conf = tf.layers.conv2d(h_conv4_3_norm, num_priors*num_classes, [3, 3], padding="same", name=name, activation=None)
        
        ### flatten conf
        h_conv4_3_norm_mbox_conf_flat = tf.layers.Flatten()(h_conv4_3_norm_mbox_conf)
        ## priorbox
        priorbox = PriorBox(img_size, 30.0, aspect_ratios=[1/2., 2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
        h_conv4_3_norm_mbox_priorbox = priorbox(h_conv4_3_norm)
        
        ########################
        # prediction from fc7###
        ########################
        num_priors = 6
        ## loc
        h_fc7_mbox_loc = tf.layers.conv2d(h_fc7, num_priors*4, [3, 3], padding="same", name="fc7_mbox_loc", activation=None)
        ## flatten: y = tf.layers.Flatten()(x)
        h_fc7_mbox_loc_flat = tf.layers.Flatten()(h_fc7_mbox_loc)
        ## conf
        name = "fc7_mbox_conf_{}".format(num_classes)
        h_fc7_mbox_conf = tf.layers.conv2d(h_fc7, num_priors*num_classes, [3, 3], padding="same", name=name, activation=None)
        ## flatten: y = tf.layers.Flatten()(x)
        h_fc7_mbox_conf_flat = tf.layers.Flatten()(h_fc7_mbox_conf)
        ## priorbox
        priorbox = PriorBox(img_size, 60.0, max_size=114, aspect_ratios=[1/2., 2, 3], variances=[0.1, 0.1, 0.2, 0.2], name="fc7_mbox_priorbox")
        h_fc7_mbox_priorbox = priorbox(h_fc7)
        
        ##########################
        # prediction from conv6_2#
        ##########################
        num_priors = 6
        ## loc
        h_conv6_2_mbox_loc = tf.layers.conv2d(h_conv6_2, num_priors*4, [3, 3], padding="same", name="conv6_2_mbox_loc", activation=None)
        ### flatten
        h_conv6_2_mbox_loc_flat = tf.layers.Flatten()(h_conv6_2_mbox_loc)
        ## conf
        name = "conv6_2_mbox_conf_{}".format(num_classes)
        h_conv6_2_mbox_conf = tf.layers.conv2d(h_conv6_2, num_priors*num_classes, [3, 3], padding="same", name=name, activation=None)
        ### flatten
        h_conv6_2_mbox_conf_flat = tf.layers.Flatten()(h_conv6_2_mbox_conf)
        ## priorbox
        priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[1/2., 2, 3], variances=[0.1,0.1,0.2,0.2], name="conv6_2_mbox_priorbox")
        h_conv6_2_mbox_priorbox = priorbox(h_conv6_2)
        
        ##########################
        # prediction from conv7_2#
        ##########################
        num_priors = 6
        ## loc
        h_conv7_2_mbox_loc = tf.layers.conv2d(h_conv7_2, num_priors*4, [3, 3], padding="same", name="conv7_2_mbox_loc", activation=None)
        ### flatten: y = tf.layers.Flatten()(x)
        h_conv7_2_mbox_loc_flat = tf.layers.Flatten()(h_conv7_2_mbox_loc)
        ## conf
        name = "conv7_2_mbox_conf_{}".format(num_classes)
        h_conv7_2_mbox_conf = tf.layers.conv2d(h_conv7_2, num_priors*num_classes, [3, 3], padding="same", name=name, activation=None)
        ### flatten: y = tf.layers.Flatten()(x)
        h_conv7_2_mbox_conf_flat = tf.layers.Flatten()(h_conv7_2_mbox_conf)
        ## priorbox
        priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name="conv7_2_mbox_priorbox")
        h_conv7_2_mbox_priorbox = priorbox(h_conv7_2)
        
        ##########################
        # prediction from conv8_2#
        ##########################
        num_priors = 6
        ## loc
        h_conv8_2_mbox_loc = tf.layers.conv2d(h_conv8_2, num_priors*4, [3, 3], padding="same", name="conv8_2_mbox_loc", activation=None)
        ## flatten
        h_conv8_2_mbox_loc_flat = tf.layers.Flatten()(h_conv8_2_mbox_loc)
        ## conf
        name = "conv8_2_mbox_conf_{}".format(num_classes)
        h_conv8_2_mbox_conf = tf.layers.conv2d(h_conv8_2, num_priors*num_classes, [3, 3], padding="same", name=name, activation=None)
        ## flatten
        h_conv8_2_mbox_conf_flat = tf.layers.Flatten()(h_conv8_2_mbox_conf)
        ## priorbox
        priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2], name="conv8_2_mbox_priorbox")
        h_conv8_2_mbox_priorbox = priorbox(h_conv8_2)
        
        #########################
        # prediction from pool6 #
        #########################
        num_priors = 6
        ## loc dense
        h_pool6_mbox_loc_flat = tf.layers.dense(h_pool6, num_priors*4, name="pool6_mbox_loc_flat", activation=None)
        ## conf
        name = "pool6_mbox_conf_flat_{}".format(num_classes)
        h_pool6_mbox_conf_flat = tf.layers.dense(h_pool6, num_priors*num_classes, name=name, activation=None)
        ## priorbox
        priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1,0.1,0.2,0.2], name="conv8_2_mbox_priorbox")
        ### reshape h_pool6
        target_shape = (1,1,256)
        h_pool6_reshaped = tf.keras.layers.Reshape(target_shape, name="pool6_reshaped")(h_pool6)
        h_pool6_mbox_priorbox = priorbox(h_pool6_reshaped)
        
        ##########################
        # Gather all predictions #
        ##########################
        h_mbox_loc = tf.concat( [h_conv4_3_norm_mbox_loc_flat, 
                                 h_fc7_mbox_loc_flat, 
                                 h_conv6_2_mbox_loc_flat, 
                                 h_conv7_2_mbox_loc_flat, 
                                 h_conv8_2_mbox_loc_flat, 
                                 h_pool6_mbox_loc_flat], axis = 1, name = "mbox_loc")
        h_mbox_conf = tf.concat( [h_conv4_3_norm_mbox_conf_flat, 
                                 h_fc7_mbox_conf_flat, 
                                 h_conv6_2_mbox_conf_flat, 
                                 h_conv7_2_mbox_conf_flat, 
                                 h_conv8_2_mbox_conf_flat, 
                                 h_pool6_mbox_conf_flat], axis = 1, name = "mbox_conf")
        h_mbox_priorbox = tf.concat( [h_conv4_3_norm_mbox_priorbox, 
                                 h_fc7_mbox_priorbox, 
                                 h_conv6_2_mbox_priorbox, 
                                 h_conv7_2_mbox_priorbox, 
                                 h_conv8_2_mbox_priorbox, 
                                 h_pool6_mbox_priorbox], axis = 1, name = "mbox_priorbox")
        
        
        # count number of prior bboxs
        num_boxes = tf.shape(h_mbox_loc)[-1] // 4
        h_mbox_loc = tf.keras.layers.Reshape((num_boxes, 4), name="mbox_loc_final")(h_mbox_loc)
        h_mbox_conf = tf.keras.layers.Reshape((num_boxes, num_classes), name="mbox_conf_logits")(h_mbox_conf)
        h_mbox_conf = tf.nn.softmax(h_mbox_conf, name="mbox_conf_final")
        
        # final prediction
        predictions = tf.concat([h_mbox_loc, h_mbox_conf, h_mbox_priorbox], axis=2, name="predictions")
        
        
        if PRIORS:
            return predictions, h_mbox_priorbox
        else:
            return predictions
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        