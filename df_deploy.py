import cv2
from gluoncv import utils
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import nd
from df_dataset import DF_Detection
# from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
from gluoncv import utils
from gluoncv.utils import viz
from matplotlib import pyplot as plt

gpus = "0"
utils.random.seed(233)
width, height = 512, 512  # suppose we use 512 as base training size
batch_size = 2
num_workers = 2

if __name__ == '__main__':
    root = "F:\DF"
    ctx = [mx.gpu(int(i)) for i in gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    test_dataset = DF_Detection(root, label_name='rect_val.csv')
    print('Validation images:', len(test_dataset))

    net = model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=test_dataset.classes)
    net.load_parameters("train_df_file/ssd_512_resnet50_v1_DF_best.params")
    net.collect_params().reset_ctx(ctx)

    batchify_fn = Tuple(Stack(), Stack())
    test_loader = DataLoader(test_dataset.transform(SSDDefaultValTransform(width, height)),
                            batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)

    net.set_nms(nms_thresh=0.45, nms_topk=100)
    for i, batch in enumerate(test_loader):
        # if i > 0:
        #     break
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        # data = batch[0]
        # label = batch[1]
        print('data:', batch[0].shape)
        print('class targets:', batch[0].shape)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []

        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)

            # det_ids.append(ids)
            # det_scores.append(scores)
            # # clip to image size
            # det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # # split ground truths
            # gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            # gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))

            img = x[0].transpose((1, 2, 0)).as_in_context(mx.cpu())
            img = img * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
            # train_image2 = (train_image2 * 255).clip(0, 255)
            img = (img * 255).clip(0, 255)
            show_img = img.asnumpy()
            # ids_cpu = ids
            # scores_cpu = scores
            # bboxes_cpu = bboxes
            # print(ids_cpu[0][0][0])
            cv2.imshow("sdfasdf", show_img)
            cv2.waitKey(0)
            # ax = viz.plot_bbox(train_image2.asnumpy(), bboxes[:, :4],
            #                    labels=y[:, 4:5],
            #                    class_names=test_dataset.classes)
            # plt.show()




    # img = batch[0][0].transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    # show_img = img.asnumpy()
    # for i in range(100):
    #     cpubboxs = bboxes[0]
    #     cpubbox = cpubboxs[i].asnumpy()
    #     show_img = cv2.rectangle(show_img, (cpubbox[1], cpubbox[0]), (cpubbox[3], cpubbox[2]), (0, 255, 255), 1)
    # cv2.imshow("img", show_img)
    # cv2.waitKey(0)


    # from matplotlib import pyplot as plt
    # from gluoncv.utils import viz
    # for i, batch in enumerate(train_loader):
    #     if i > 0:
    #         break
    #     print('data:', batch[0].shape)
    #     print('class targets:', batch[1].shape)
    #     print('box targets:', batch[2].shape)
    #     data = batch[0][0]
    #     img = data.transpose(
    #         (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
    #     # train_image2 = (train_image2 * 255).clip(0, 255)
    #     train_image2 = (train_image2 * 255).clip(0, 255)
    #     ax = viz.plot_bbox(train_image2.asnumpy(), batch[1][:, :4],
    #                        labels=batch[2][:, ],
    #                        class_names=train_dataset.classes)
    #     plt.show()




# if __name__ == '__main__':
#     root = "E:\DataFountain\TSR"
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
#     # train_loader, val_loader = get_dataloader(net, train_dataset, val_dataset, width, height, num_workers)
#
#
#     with autograd.train_mode():
#         _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
#
#     # behavior of batchify_fn: stack images, and pad labels
#     train_transformed = SSDDefaultTrainTransform(width, height, anchors)
#
#     train_image, train_label = train_dataset[0]
#
#
#     train_image2, cids, train_label2 = train_transformed(train_image, train_label)
#
#
#     train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
#     # train_image2 = (train_image2 * 255).clip(0, 255)
#     cvimg = train_image2.asnumpy()
#
#
#     # cvimg = cv2.rectangle(cvimg, (train_label2[0][0], train_label2[0][1]), (train_label2[0][2], train_label2[0][3]), (0, 255, 0), 1)
#     cvimg = cv2.putText(cvimg, 's', (train_label2[0][2].asscalar(), train_label2[0][3].asscalar()), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 4)
#     cv2.imshow("dsfa", cvimg)
#     cv2.waitKey(0)
#
#
#
#
#     batchify_fn = Tuple(Stack(), Stack(), Stack())
#     train_loader = DataLoader(train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
#                               batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
#
#     val_loader = DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
#                             batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep', num_workers=num_workers)



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



#
# import os
# import cv2
# import time
# import numpy as np
# import mxnet as mx
# from gluoncv import model_zoo
# from mxnet import gluon
#
# try:
#     import xml.etree.cElementTree as ET
# except ImportError:
#     import xml.etree.ElementTree as ET
#
#
# class TMRI_slagcar_Detection(object):
#     def __init__(self, net_params: str, ctx, threshold):
#         if not os.path.exists(net_params):
#             print(print("{} not exists".format(net_params)))
#         self._ctx = ctx
#         self._threshold = threshold
#         self._net = model_zoo.get_model('ssd_512_resnet50_v1_custom', classes=['slagcar', 'other'], pretrained_base=False)
#         self._net.load_parameters(net_params)
#         self._net.collect_params().reset_ctx(ctx)
#         self._net.set_nms(nms_thresh=0.45, nms_topk=400)
#         self._net.hybridize()
#
#     def _preprocess(self, img):
#         def to_inference(img, ctx):
#             x = mx.nd.array(img)
#             x = mx.nd.image.to_tensor(x)
#             x = mx.nd.image.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#             return x.expand_dims(axis=0).as_in_context(ctx)
#
#         img = cv2.resize(img, (512, 512))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         x = to_inference(img, self._ctx)
#
#         # 水平翻转
#         z = cv2.flip(img, 1)
#         z = to_inference(z, self._ctx)
#         return x, z
#
#     def predict(self, img, ensemble=False, area_filter=False):
#         h, w, _ = img.shape
#         # 图片预处理，x未作处理的图片，z图片水平翻转
#         x, z = self._preprocess(img)
#         ids, scores, bboxes = self._net(x)
#
#         if ensemble:
#             result_0 = mx.nd.concat(*[ids, scores, bboxes], dim=-1)
#             ids, scores, bboxes = self._net(z)
#
#             # 水平翻转bboxes
#             temp = 512 - bboxes[:, :, 2]
#             bboxes[:, :, 2] = 512 - bboxes[:, :, 0]
#             bboxes[:, :, 0] = temp
#             result_1 = mx.nd.concat(*[ids, scores, bboxes], dim=-1)
#
#             result = mx.nd.concat(*[result_0, result_1], dim=1)
#             result = mx.nd.contrib.box_nms(
#                 result, overlap_thresh=0.45, topk=100, valid_thresh=0.01,
#                 id_index=0, score_index=1, coord_start=2, force_suppress=False)
#
#             result = mx.nd.slice_axis(result, axis=1, begin=0, end=100)
#             ids = mx.nd.slice_axis(result, axis=2, begin=0, end=1)
#             scores = mx.nd.slice_axis(result, axis=2, begin=1, end=2)
#             bboxes = mx.nd.slice_axis(result, axis=2, begin=2, end=6)
#         # 取出需要的类
#         ids = ids.asnumpy().squeeze()
#         valid = (ids == 0)
#         ids = ids[valid]
#         scores = scores.asnumpy().squeeze()[valid]
#         bboxes = bboxes.asnumpy().squeeze()[valid, :]
#
#         if area_filter:
#             area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
#             valid = np.logical_and(area.flat < 512 * 512 * 0.41, area.flat > 512 * 512 * 0.00829)
#             ids = ids[valid]
#             scores = scores[valid]
#             bboxes = bboxes[valid, :]
#         valid = np.where(scores >= self._threshold)[0]
#         ids = ids[valid]
#         bboxes = bboxes[valid, :]
#         scores = scores[valid]
#
#         return ids, scores, bboxes / 512 * [w, h, w, h]
#
#
# def _load_groundtruth(self, root):
#     """Parse xml file and return labels."""
#     items = []
#     for files in os.listdir(self._anno_root):
#
#         anno_path = os.path.join(self._anno_root, files)
#         root = ET.parse(anno_path).getroot()
#         image_name = root.find('filename').text
#         size = root.find('size')
#         width = float(size.find('width').text)
#         height = float(size.find('height').text)
#         label = []
#         for obj in root.iter('object'):
#             difficult = int(obj.find('difficult').text)
#             cls_name = obj.find('name').text.strip().lower()
#             if cls_name not in self.classes:
#                 continue
#             cls_id = self.index_map[cls_name]
#             xml_box = obj.find('bndbox')
#             xmin = float(xml_box.find('xmin').text)
#             ymin = float(xml_box.find('ymin').text)
#             xmax = float(xml_box.find('xmax').text)
#             ymax = float(xml_box.find('ymax').text)
#             try:
#                 self._validate_label(xmin, ymin, xmax, ymax, width, height)
#             except AssertionError as e:
#                 raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
#             label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
#         items.append([image_name, np.array(label)])
#     return items
#
#
# if __name__ == "__main__":
#
#     start = time.time()
#     params = "train_slagcar_file_old/ssd_512_resnet50_v1_slagcar_0199_0.6586.params"
#     image_root = "/media/zatuche/train2/JPEGImages"
#     anno_root = '/media/zatuche/train2/Annotations'
#     ctx = mx.gpu()
#     threshold = 0.9040
#     net = TMRI_slagcar_Detection(params, ctx, threshold)
#     counter = 0
#
#     for file in os.listdir(anno_root):
#
#         anno_path = os.path.join(anno_root, file)
#         root = ET.parse(anno_path).getroot()
#         image_name = root.find('filename').text
#         image_path = os.path.join(image_root, image_name)
#         img = cv2.imread(image_path)
#         ids, scores, bboxes = net.predict(img)
#         print("shape:", ids.shape, scores.shape, bboxes.shape)
#         print("ids:", ids)
#         print("scores:", scores)
#         print("bboxes:", bboxes)
#         counter += 1
#         if counter > 10:
#             break
#         height, width = img.shape[:2]
#         for i in range(0, bboxes.shape[0]):
#             cv2.rectangle(img, (int(bboxes[i, 0]), int(bboxes[i, 1])), (int(bboxes[i, 2]), int(bboxes[i, 3])), (0, 0, 255), thickness=4)
#             cv2.putText(img, str(scores[i]), (int(bboxes[i, 0]), int(bboxes[i, 3])), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), thickness=2)
#             # print("ratio: ", (bboxes[i, 3] - bboxes[i, 1]) * (bboxes[i, 2] - bboxes[i, 0]) / width / height)
#
#         for obj in root.iter('object'):
#             difficult = int(obj.find('difficult').text)
#             xml_box = obj.find('bndbox')
#             xmin = float(xml_box.find('xmin').text)
#             ymin = float(xml_box.find('ymin').text)
#             xmax = float(xml_box.find('xmax').text)
#             ymax = float(xml_box.find('ymax').text)
#             cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), thickness=4)
#
#         img2 = cv2.resize(img, (int(width / 3), int(height / 3)))
#         cv2.imwrite("testImage/test-{}.jpg".format(counter), img2)
#     stop = time.time()
#     print("test time:", (stop - start) / counter)




