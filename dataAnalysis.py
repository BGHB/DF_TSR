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


from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from df_dataset import DF_Detection
root = "/media/handewei/新材料/DF"
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd

# ##############################################################################
# width, height = 512, 512  # suppose we use 512 as base training size
# train_transform = presets.ssd.SSDDefaultTrainTransform(width, height)
# val_transform = presets.ssd.SSDDefaultValTransform(width, height)
#
#
#
#
# batch_size = 2  # for tutorial, we use smaller batch-size
# # you can make it larger(if your CPU has more cores) to accelerate data loading
# num_workers = 0
#
# train_dataset = DF_Detection(root, label_name='train_label.csv')
# val_dataset = DF_Detection(root, label_name='val_label.csv')
# print('Training images:', len(train_dataset))
# print('Validation images:', len(val_dataset))
#
#
# # behavior of batchify_fn: stack images, and pad labels
# batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
# train_loader = DataLoader(
#     train_dataset.transform(train_transform),
#     batch_size,
#     shuffle=True,
#     batchify_fn=batchify_fn,
#     last_batch='rollover',
#     num_workers=num_workers)
# val_loader = DataLoader(
#     val_dataset.transform(val_transform),
#     batch_size,
#     shuffle=False,
#     batchify_fn=batchify_fn,
#     last_batch='keep',
#     num_workers=num_workers)
#
# for ib, batch in enumerate(train_loader):
#     if ib > 3:
#         break
#     print('data:', batch[0].shape, 'label:', batch[1].shape)

from gluoncv import model_zoo

net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
print(net)

##############################################################################
# SSD network is a HybridBlock as mentioned before. You can call it with
# an input as:
import mxnet as mx

x = mx.nd.zeros(shape=(1, 3, 512, 512))
net.initialize()
cids, scores, bboxes = net(x)
