import numpy as np
import tensorflow as tf

class AtrousConvolution2D(object):
    def __init__(self, cfg, rate=(6, 6), padding="SAME", name="atrous_conv2d"):
        """
        filters, a 4D tensor
        """
        super().__init__()
        #[h, w, in_chls, out_chls] = cfg
        init_filters = np.random.normal(loc=0.0, scale=0.05, size=cfg)
        self.filters = tf.Variable(init_filters, dtype=tf.float32)
        
        self.rate = rate
        self.padding = padding
        self.name = name
        
    def __call__(self, value):
        filters, rate, padding, name = self.filters, self.rate, self.padding, self.name
        return tf.nn.atrous_conv2d(value, filters, rate, padding, name=name)

"""
class Normalize(object):
    def __init__(self, scale, name):
        #if K.image_dim_ordering() == 'tf':
        self.axis = 3
        #else:
        #    self.axis = 1
        self.scale = scale
        self.name = name
        super().__init__()
        # initialize variables
        self.build()
    
    def build(self, input_shape):
        #self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        #self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        #self.trainable_weights = [self.gamma]
        self.gamma = tf.Variable(init_gamma, name='{}_gamma'.format(self.name), trainable=True)

    def call(self, x, mask=None):
        #output = K.l2_normalize(x, self.axis)
        output = tf.nn.l2_normalize(x, self.axis)
        output *= self.gamma
        return output
"""    
   
class Normalize(object):
    def __init__(self, scale, name=None):
        #if K.image_dim_ordering() == 'tf':
        self.axis = 3
        #else:
        #    self.axis = 1
        self.scale = scale
        self.name = name
        
        #print(self)
        #print(Normalize)
        #print("hhh")
        
        #super(Normalize, self).__init__(**kwargs)
        
        #self.input_spec = [InputSpec(shape=input_shape)]
        shape = 128#(input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape).astype("float32")
        #self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.gamma = tf.Variable(init_gamma, name='{}_gamma'.format(self.name), trainable=True)

    def build(self, input_shape):
        pass

    def __call__(self, x, mask=None):
        output = tf.nn.l2_normalize(x, self.axis)
        
        #print(output)
        #print(self.gamma)
        
        output *= self.gamma
        return output    

    
    
class PriorBox(object):
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, name="priorbox"):
        self.name = name
        
        #if K.image_dim_ordering() == 'tf':
        self.waxis = 2
        self.haxis = 1
        #else:
        #    self.waxis = 3
        #    self.haxis = 2
        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        #super(PriorBox, self).__init__()
    
    def get_output_shape_for(self, input_shape):
        num_priors_ = len(self.aspect_ratios)
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        num_boxes = num_priors_ * layer_width * layer_height
        return (input_shape[0], num_boxes, 8)

    def __call__(self, x, mask=None):
        #if hasattr(x, '_keras_shape'):
        #    input_shape = x._keras_shape
        #elif hasattr(K, 'int_shape'):
        #    input_shape = K.int_shape(x)
        
        # input x will be a tensorflow tensor
        input_shape = x.get_shape().as_list()#tf.shape(x)
        
        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]
        img_width = self.img_size[0]
        img_height = self.img_size[1]
        # define prior boxes shapes
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4) # prior_boxes: [num_boxes, x1 x2 y1 y2]
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')
        prior_boxes = np.concatenate((prior_boxes, variances), axis=1).astype("float32")
        #prior_boxes_tensor = K.expand_dims(K.variable(prior_boxes), 0)
        prior_boxes_tensor = tf.expand_dims(tf.Variable(prior_boxes, trainable=False, name=self.name), 0, name=self.name+"_tensor")
        
        #if K.backend() == 'tensorflow':
        pattern = [tf.shape(x)[0], 1, 1]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)
        #elif K.backend() == 'theano':
            #TODO
        #    pass
        
        return prior_boxes_tensor