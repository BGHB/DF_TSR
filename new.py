##########################################################
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


gpus = "0"
utils.random.seed(233)
width, height = 512, 512  # suppose we use 512 as base training size
batch_size = 2
num_workers = 4
root = "/media/handewei/新材料/DF"


def get_dataloader(net, train_dataset, val_dataset, width, height, num_workers):
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    # behavior of batchify_fn: stack images, and pad labels

    batchify_fn_t = Tuple(Stack(), Stack(), Stack())
    train_loader = DataLoader(train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
                              batch_size, shuffle=False, batchify_fn=batchify_fn_t, last_batch='rollover', num_workers=num_workers)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
                            batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


mbox_loss = SSDMultiBoxLoss()
# if __name__ == '__main__':
#     ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
#     ctx = ctx if ctx else [mx.cpu()]
#
#     train_dataset = DF_Detection(root, label_name='train_label.csv')
#     val_dataset = DF_Detection(root, label_name='val_label.csv')
#     print('Training images:', len(train_dataset))
#     print('Validation images:', len(val_dataset))
#
#     net = model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=train_dataset.classes)
#     net.initialize(force_reinit=True)
#
#     train_loader, val_loader = get_dataloader(net, train_dataset, val_dataset, width, height, num_workers)
#
#     # train
#     trainer = gluon.Trainer(net.collect_params(), 'sgd',
#         {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
#
#     for ib, batch in enumerate(train_loader):
#         if ib > 0:
#             break
#         print('data:', batch[0].shape)
#         print('class targets:', batch[1].shape)
#         print('box targets:', batch[2].shape)
#         with autograd.record():
#             cls_pred, box_pred, anchors = net(batch[0])
#             sum_loss, cls_loss, box_loss = mbox_loss(
#                 cls_pred, box_pred, batch[1], batch[2])
#             # some standard gluon training steps:
#             # autograd.backward(sum_loss)
#             # trainer.step(1)

class SSDDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2),
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        from gluoncv.model_zoo.ssd.target import SSDTargetGenerator
        self._target_generator = SSDTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        # img = experimental.image.random_color_distort(src)
        img, bbox = src, label

        # resize with random interpolation
        h, w, _ = img.shape
        img = mx.image.imresize(img, self._width, self._height)
        x_scale = self._height / h
        y_scale = self._width / w
        bbox[:, 1] = y_scale * label[:, 1]
        bbox[:, 3] = y_scale * label[:, 3]
        bbox[:, 0] = x_scale * label[:, 0]
        bbox[:, 2] = x_scale * label[:, 2]

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0]



if __name__ == '__main__':
    ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    train_dataset = DF_Detection(root, label_name='train_label.csv')
    val_dataset = DF_Detection(root, label_name='val_label.csv')
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))

    net = model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=train_dataset.classes)
    net.initialize(force_reinit=True)

    # train_loader, val_loader = get_dataloader(net, train_dataset, val_dataset, width, height, num_workers)


    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))

    # behavior of batchify_fn: stack images, and pad labels
    train_transformed = SSDDefaultTrainTransform(width, height, anchors)

    train_image, train_label = train_dataset[0]


    train_image2, cids, train_label2 = train_transformed(train_image, train_label)


    train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    # train_image2 = (train_image2 * 255).clip(0, 255)
    cvimg = train_image2.asnumpy()
    cvimg = cv2.rectangle(cvimg, (train_label2[0][0], train_label2[0][1]), (train_label2[0][2], train_label2[0][3]), (0, 255, 0), 1)
    cvimg = cv2.putText(cvimg, str(cids[0][0]), (train_label2[0][2], train_label2[0][3]), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 8)
    cv2.imshow("dsfa", cvimg)
    cv2.waitKey(0)




    batchify_fn_t = Tuple(Stack(), Stack(), Stack())
    train_loader = DataLoader(train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
                              batch_size, shuffle=False, batchify_fn=batchify_fn_t, last_batch='rollover', num_workers=num_workers)
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
                            batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)




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



