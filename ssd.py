import time
import numpy as np
import cv2
import csv
import glob
import random

import mxnet as mx
import mxnet.image as image
from mxnet import nd
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.contrib.ndarray import MultiBoxTarget
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.contrib.ndarray import MultiBoxDetection

print("all module imported")

NUM_CLASS = 43
BATCH_SIZE = 15
DATA_SHAPE = 256

train_data = mx.image.ImageIter(
   batch_size=BATCH_SIZE, label_width = 1,
   data_shape=(3, DATA_SHAPE, DATA_SHAPE),  
   path_imgrec='./dataset/dataset.rec',  
   path_imgidx='./dataset/dataset.idx',  #help shuffle performance
   shuffle=True)
test_data = mx.image.ImageIter(  
   batch_size=BATCH_SIZE, label_width = 1,
   data_shape=(3, DATA_SHAPE, DATA_SHAPE),  
   path_imgrec='./dataset/dataset.rec',  
   path_imgidx='./dataset/dataset.idx',  #help shuffle performance
   shuffle=True)
print("image loaded")

paths = glob.glob("dataset/scene-jpg/*.jpg")
labels = nd.zeros((len(paths), NUM_CLASS+1, 5)) -1.
gts = open("dataset/gt1.txt",'r').read().split('\n')[:-1]
for gt in gts:
    line = gt.split(";")
    idx = int(line[0].split(".")[0])
    minx = float(line[1])
    miny = float(line[2])
    maxx = float(line[3])
    maxy = float(line[4])
    label = float(line[5])
    labels[idx][int(label)] = [label, minx, miny, maxx, maxy]
print("label loaded")

signname_file = "dataset/signnames.csv"
with open(signname_file) as f:
    f.readline() # skip the headers
    signnames = [row[1] for row in csv.reader(f)]
print("sign names loaded")


def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer 
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

def flatten_prediction(pred):
    return nd.flatten(nd.transpose(pred, axes=(0, 2, 3, 1)))

def concat_predictions(preds):
    return nd.concat(*preds, dim=1)

def body():
    """return the body network"""
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out

def toy_ssd_model(num_anchors, num_classes):
    """return SSD modules"""
    downsamples = nn.Sequential()
    class_preds = nn.Sequential()
    box_preds = nn.Sequential()

    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    downsamples.add(down_sample(128))
    
    for scale in range(5):
        class_preds.add(class_predictor(num_anchors, num_classes))
        box_preds.add(box_predictor(num_anchors))
    
    return body(), downsamples, class_preds, box_preds

def toy_ssd_forward(x, body, downsamples, class_preds, box_preds, sizes, ratios):                
    # extract feature with the body network        
    x = body(x)
        
    # for each scale, add anchors, box and class predictions,
    # then compute the input to next scale 
    default_anchors = []
    predicted_boxes = []  
    predicted_classes = []
                        
    for i in range(5):
        default_anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        predicted_boxes.append(flatten_prediction(box_preds[i](x)))
        predicted_classes.append(flatten_prediction(class_preds[i](x)))
        if i < 3:
            x = downsamples[i](x)
        elif i == 3:
            # simply use the pooling layer
            x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))

    return default_anchors, predicted_classes, predicted_boxes

class ToySSD(gluon.Block):
    def __init__(self, num_classes, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes for 4 feature scales
        self.anchor_sizes = [[.1, .11], [.12, .15], [.18, .2], [.22, .25], [.27, .3]]
        # anchor box ratios for 4 feature scales
        self.anchor_ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes

        with self.name_scope():
            self.body, self.downsamples, self.class_preds, self.box_preds = toy_ssd_model(4, num_classes)
            
    def forward(self, x):
        default_anchors, predicted_classes, predicted_boxes = toy_ssd_forward(x, self.body, self.downsamples,
            self.class_preds, self.box_preds, self.anchor_sizes, self.anchor_ratios)
        # we want to concatenate anchors, class predictions, box predictions from different layers
        anchors = concat_predictions(default_anchors)
        box_preds = concat_predictions(predicted_boxes)
        class_preds = concat_predictions(predicted_classes)
        # it is better to have class predictions reshaped for softmax computation
        class_preds = nd.reshape(class_preds, shape=(0, -1, self.num_classes + 1))
        
        return anchors, class_preds, box_preds

print("model done")

def training_targets(default_anchors, class_predicts, labels):
    class_predicts = nd.transpose(class_predicts, axes=(0, 2, 1))
    z = MultiBoxTarget(*[default_anchors, labels, class_predicts])
    box_target = z[0]  # box offset target for (x, y, width, height)
    box_mask = z[1]  # mask is used to ignore box offsets we don't want to penalize, e.g. negative samples
    cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
    return box_target, box_mask, cls_target

class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
    
    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pt = F.pick(output, label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pt) ** self._gamma) * F.log(pt)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
    
    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return F.mean(loss, self._batch_axis, exclude=True)


cls_loss = FocalLoss()
box_loss = SmoothL1Loss()
cls_metric = mx.metric.Accuracy()
box_metric = mx.metric.MAE()
ctx = mx.gpu()

net = ToySSD(NUM_CLASS)
net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
print(net.num_classes)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

log_interval = 20
start_epoch = 0
epochs = 20

print("start train")
for epoch in range(start_epoch, epochs):
    # reset iterator and tick
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(train_data):
        btic = time.time()
        # record gradients
        with ag.record():
            x = batch.data[0].as_in_context(ctx) - 128.
            batch_label = batch.label[0].asnumpy().astype(int)
            y = labels[batch_label].as_in_context(ctx)
            default_anchors, class_predictions, box_predictions = net(x)
            box_target, box_mask, cls_target = training_targets(default_anchors, class_predictions, y)
            # losses
            loss1 = cls_loss(class_predictions, cls_target)
            loss2 = box_loss(box_predictions, box_target, box_mask)
            # sum all losses
            loss = loss1 + loss2
            # backpropagate
            loss.backward()
        # apply 
        trainer.step(BATCH_SIZE)
        # update metrics
        cls_metric.update([cls_target], [nd.transpose(class_predictions, (0, 2, 1))])
        box_metric.update([box_target], [box_predictions * box_mask])
        if (i + 1) % log_interval == 0:
            name1, val1 = cls_metric.get()
            name2, val2 = box_metric.get()
            print('[Epoch %d Batch %d] speed: %d samples/s, train: %s=%f, %s=%f' 
                  %(epoch ,i, BATCH_SIZE/(time.time()-btic), name1, val1, name2, val2))
    
    # end of epoch logging
    name1, val1 = cls_metric.get()
    name2, val2 = box_metric.get()
    print('[Epoch %d] train: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
    print('[Epoch %d] time: %f'%(epoch, time.time()-tic))
    
# we can save the trained parameters to disk
net.save_params('ssd_%d.params' % epochs)
print("end train")


