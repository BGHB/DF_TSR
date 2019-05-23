import cv2
from gluoncv import utils
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import autograd
from mxnet import nd
from df_dataset import DF_Detection
# from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from gluoncv.loss import SSDMultiBoxLoss
from df_train import TrainTransform

gpus = "0"
utils.random.seed(233)
width, height = 512, 512  # suppose we use 512 as base training size
batch_size = 2
num_workers = 4

def get_dataloader(net, train_dataset, val_dataset, width, height, num_workers):
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn_t = Tuple(Stack(), Stack(), Stack())
    train_loader = DataLoader(train_dataset.transform(TrainTransform(width, height, anchors)),
                              batch_size, shuffle=False, batchify_fn=batchify_fn_t, last_batch='rollover', num_workers=num_workers)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
                            batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    root = "E:\DataFountain\TSR"
    ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    train_dataset = DF_Detection(root, label_name='sub_train_label.csv')
    val_dataset = DF_Detection(root, label_name='sub_val_label.csv')
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))

    net = model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=train_dataset.classes)
    net.initialize(force_reinit=True)

    # train_loader, val_loader = get_dataloader(net, train_dataset, val_dataset, width, height, num_workers)

    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    train_transformed = TrainTransform(width, height, anchors)
    # behavior of batchify_fn: stack images, and pad labels
    batchify_fn = Tuple(Stack(), Stack(), Stack())
    train_loader = DataLoader(train_dataset.transform(train_transformed),
                              batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    # batchify_fn = Tuple(Stack(), Stack())
    # val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
    #                         batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)
    from matplotlib import pyplot as plt
    from gluoncv.utils import viz
    for i, batch in enumerate(train_loader):
        if i > 0:
            break
        print('data:', batch[0].shape)
        print('class targets:', batch[1].shape)
        print('box targets:', batch[2].shape)
        data = batch[0][0]
        img = data.transpose(
            (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
        # train_image2 = (train_image2 * 255).clip(0, 255)
        train_image2 = (train_image2 * 255).clip(0, 255)
        ax = viz.plot_bbox(train_image2.asnumpy(), batch[1][:, :4],
                           labels=batch[2][:, ],
                           class_names=train_dataset.classes)
        plt.show()
        # img = data.transpose(1, 2, 0)
        # img = img.asnumpy()
        # cv2.imshow("win", img)
        # cv2.waitKey(0)
        # train_image, train_label = train_dataset[0]
        #
        # train_image2, cids, train_label2 = train_transformed(train_image, train_label)
        #
        # train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
        # # train_image2 = (train_image2 * 255).clip(0, 255)
        # cvimg = train_image2.asnumpy()
        #
        # # cvimg = cv2.rectangle(cvimg, (train_label2[0][0], train_label2[0][1]), (train_label2[0][2], train_label2[0][3]), (0, 255, 0), 1)
        # cvimg = cv2.putText(cvimg, 's', (train_label2[0][2].asscalar(), train_label2[0][3].asscalar()), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)
        # cv2.imshow("dsfa", cvimg)
        # cv2.waitKey(0)




    # batchify_fn = Tuple(Stack(), Stack(), Stack())
    # train_loader = DataLoader(train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
    #                           batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    #
    # val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
    #                         batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)




    # # train
    # trainer = gluon.Trainer(net.collect_params(), 'sgd',
    #     {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
    #
    # for ib, batch in enumerate(train_loader):
    #     if ib > 0:
    #         break
    #     print('data:', batch[0].shape)
    #     print('class targets:', batch[1].shape)
    #     print('box targets:', batch[2].shape)
    #     with autograd.record():
    #         cls_pred, box_pred, anchors = net(batch[0])
    #         sum_loss, cls_loss, box_loss = mbox_loss(
    #             cls_pred, box_pred, batch[1], batch[2])
    #         # some standard gluon training steps:
    #         # autograd.backward(sum_loss)
    #         # trainer.step(1)



##############################################################################
# apply transforms to train image
# train_image2, train_label2 = train_transform(train_image, train_label)
# print('tensor shape:', train_image2.shape)

# ##############################################################################
# # Images in tensor are distorted because they no longer sit in (0, 255) range.
# # Let's convert them back so we can see them clearly.
# train_image2 = train_image2.transpose(
#     (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
# # train_image2 = (train_image2 * 255).clip(0, 255)
# cvimg = train_image2.asnumpy()
#
# cvimg = cv2.rectangle(cvimg, (train_label2[0][0], train_label2[0][1]), (train_label2[0][2], train_label2[0][3]), (0, 255, 0), 8)
# cvimg = cv2.putText(cvimg, str(cids[0][0]), (train_label2[0][2], train_label2[0][3]), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 8)
# cv2.imshow("dsfa", cvimg)
# cv2.waitKey(0)



##########################################################
# SSD Network
# ------------------
# GluonCV's SSD implementation is a composite Gluon HybridBlock
# (which means it can be exported
# to symbol to run in C++, Scala and other language bindings.
# We will cover this usage in future tutorials).
# In terms of structure, SSD networks are composed of base feature extraction
# network, anchor generators, class predictors and bounding box offset predictors.
#
# For more details on how SSD detector works, please refer to our introductory
# [tutorial](http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)
# You can also refer to the original paper to learn more about the intuitions
# behind SSD.
#
# `Gluon Model Zoo <../../model_zoo/index.html>`__ has a lot of built-in SSD networks.
# You can load your favorite one with one simple line of code:
#
# .. hint::
#
#    To avoid downloading models in this tutorial, we set `pretrained_base=False`,
#    in practice we usually want to load pre-trained imagenet models by setting
#    `pretrained_base=True`.



##############################################################################
# SSD network is a HybridBlock as mentioned before. You can call it with
# an input as:
# import mxnet as mx
# x = mx.nd.zeros(shape=(1, 3, 512, 512))
# net.initialize()
# cids, scores, bboxes = net(x)

# ##############################################################################
# # SSD network behave differently during training mode:
# from mxnet import autograd
# with autograd.train_mode():
#     cls_preds, box_preds, anchors = net(x)
#
# ##############################################################################
# # In training mode, SSD returns three intermediate values,
# # where ``cls_preds`` are the class predictions prior to softmax,
# # ``box_preds`` are bounding box offsets with one-to-one correspondence to anchors
# # and ``anchors`` are absolute coordinates of corresponding anchors boxes, which are
# # fixed since training images use inputs of same dimensions.
#
#
# ##########################################################
# # Training targets
# # ------------------
# # Unlike a single ``SoftmaxCrossEntropyLoss`` used in image classification,
# # the loss used in SSD is more complicated.
# # Don't worry though, because we have these modules available out of the box.
# #
# # To speed up training, we let CPU to pre-compute some training targets.
# # This is especially nice when your CPU is powerful and you can use ``-j num_workers``
# # to utilize multi-core CPU.
#
# ##############################################################################
# # If we provide anchors to the training transform, it will compute
# # training targets
# from mxnet import gluon
# train_transform = presets.ssd.SSDDefaultTrainTransform(width, height, anchors)
# batchify_fn = Tuple(Stack(), Stack(), Stack())
# train_loader = DataLoader(
#     train_dataset.transform(train_transform),
#     batch_size,
#     shuffle=True,
#     batchify_fn=batchify_fn,
#     last_batch='rollover',
#     num_workers=num_workers)



