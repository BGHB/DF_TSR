# import os, cv2, csv
# import numpy as np
# from df_dataset import DF_Detection

# if __name__ == "__main__":
#     csv_file = open("/media/handewei/新材料/DF/val_label.csv")
#     lines = csv.reader(csv_file)
#     next(lines)
#     for line in lines:
#         bbox = [int(line[1]), int(line[2]), int(line[5]), int(line[6])]
#         if bbox[2] - bbox[0] > 200 or bbox[2] - bbox[0] < 18 or bbox[3] - bbox[1] < 18 or bbox[3] - bbox[1] > 200:
#             img_path = os.path.join("/media/handewei/新材料/DF/Train", line[0])
#             print(line[0])
#             if os.path.exists(img_path):
#                 cvimg = cv2.imread(img_path)
#                 cvimg = cv2.rectangle(cvimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
#                 cvimg = cv2.resize(cvimg, (1600, 900))
#                 save_path = os.path.join("/media/handewei/新材料/DF/rect", line[0])
#                 cv2.imshow("dsfa", cvimg)
#                 cv2.imwrite(save_path, cvimg)
#                 cv2.waitKey(1)


# if __name__ == "__main__":
#     hbin = np.zeros((200))
#     wbin = np.zeros((200))
#     csv_file = open("/media/handewei/新材料/DF/train_label.csv")
#     lines = csv.reader(csv_file)
#     next(lines)
#
#     for line in lines:
#         h = int(line[6]) - int(line[2])
#         w = int(line[5]) - int(line[1])
#         hbin[h] += 1
#         wbin[w] += 1
#
#     csv_file = open("/media/handewei/新材料/DF/val_label.csv")
#     lines = csv.reader(csv_file)
#     next(lines)
#
#     for line in lines:
#         h = int(line[6]) - int(line[2])
#         w = int(line[5]) - int(line[1])
#         hbin[h] += 1
#         wbin[w] += 1
#
#     print("hbin\n", hbin, sum(hbin))
#     print("wbin\n", wbin, sum(wbin))


# ##############################################################################
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from df_dataset import DF_Detection
from new import TrainTransform
root = "E:\DataFountain\TSR"
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd


width, height = 512, 512  # suppose we use 512 as base training size
train_transform = TrainTransform(width, height)
val_transform = presets.ssd.SSDDefaultValTransform(width, height)

batch_size = 2  # for tutorial, we use smaller batch-size
num_workers = 1
utils.random.seed(233)

train_dataset = DF_Detection(root, label_name='sub_train_label.csv')
# val_dataset = DF_Detection(root, label_name='val_label.csv')
print('Training images:', len(train_dataset))
# print('Validation images:', len(val_dataset))

train_image, train_label = train_dataset[0]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

##############################################################################
# Plot the image, together with the bounding box labels:
from matplotlib import pyplot as plt
from gluoncv.utils import viz

ax = viz.plot_bbox(
    train_image.asnumpy(),
    bboxes,
    labels=cids,
    class_names=train_dataset.classes)
plt.show()


##############################################################################
# apply transforms to train image
train_image2, train_label2 = train_transform(train_image, train_label)
print('tensor shape:', train_image2.shape)

##############################################################################
# Images in tensor are distorted because they no longer sit in (0, 255) range.
# Let's convert them back so we can see them clearly.
train_image2 = train_image2.transpose(
    (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
train_image2 = (train_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
                   labels=train_label2[:, 4:5],
                   class_names=train_dataset.classes)
plt.show()



###################################################################################
from gluoncv import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
print(net)
#
# ##############################################################################
# # SSD network is a HybridBlock as mentioned before. You can call it with
# # an input as:
# import mxnet as mx
# x = mx.nd.zeros(shape=(1, 3, 512, 512))
# net.initialize()
# cids, scores, bboxes = net(x)
#
# ##############################################################################
# # SSD returns three values, where ``cids`` are the class labels,
# # ``scores`` are confidence scores of each prediction,
# # and ``bboxes`` are absolute coordinates of corresponding bounding boxes.
#
# ##############################################################################
# # SSD network behave differently during training mode:
from mxnet import autograd
with autograd.train_mode():
    cls_preds, box_preds, anchors = net(x)
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

##############################################################################
# If we provide anchors to the training transform, it will compute
# training targets
from mxnet import gluon
train_transform = presets.ssd.SSDDefaultTrainTransform(width, height, anchors)
batchify_fn = Tuple(Stack(), Stack(), Stack())
train_loader = DataLoader(
    train_dataset.transform(train_transform),
    batch_size,
    shuffle=True,
    batchify_fn=batchify_fn,
    last_batch='rollover',
    num_workers=num_workers)
#
# ##############################################################################
# # Loss, Trainer and Training pipeline
# from gluoncv.loss import SSDMultiBoxLoss
# mbox_loss = SSDMultiBoxLoss()
# trainer = gluon.Trainer(
#     net.collect_params(), 'sgd',
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
#
