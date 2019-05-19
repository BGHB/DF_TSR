"""Train SSD"""
import argparse
import os
import logging
import time
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv

from df_dataset import DF_Detection
from new import TrainTransform

from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from gluoncv.utils.metrics.accuracy import Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='DF',
                        help='Training dataset.')
    parser.add_argument('--dataset-root', type=str, default="E:\DataFountain\TSR",
                        help='dataset root path')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int, default=1,
                        help='Number of data workers, \
                        you can use larger number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='10,20',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='train_df_file/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=200,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=40,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    args = parser.parse_args()
    return args

def get_dataset(root):
    train_dataset = DF_Detection(root, label_name='sub_train_label.csv')
    val_dataset = DF_Detection(root, label_name='sub_val_label.csv')
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=train_dataset.classes)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape

    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, 512, 512)))
    batchify_fn_train = Tuple(Stack(), Stack(), Stack())
    train_loader = gluon.data.DataLoader(train_dataset.transform(TrainTransform(width, height, anchors)),
                                         batch_size, shuffle=False, batchify_fn=batchify_fn_train, last_batch='rollover',
                                         num_workers=num_workers)

    batchify_fn = Tuple(Stack(), Stack())
    val_loader = gluon.data.DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
                                       batch_size, shuffle=False, batchify_fn=batchify_fn, last_batch='keep',
                                       num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch+1) % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    # net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []


        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)

            img = batch[0][0].transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
            show_img = img.asnumpy()
            for i in range(100):
                cpubboxs = bboxes[0]
                cpubbox = cpubboxs[i].asnumpy()
                show_img = cv2.rectangle(show_img, (cpubbox[1], cpubbox[0]), (cpubbox[3], cpubbox[2]), (0, 255, 255), 1)
            cv2.imshow("img", show_img)
            cv2.waitKey(0)

            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        # net.collect_params("ssd.*_expand.*|ssd.*_convpredictor.*"),
        net.collect_params(),
        'RMSProp',
        {'learning_rate': args.lr})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    batch_iter = 0
    tic = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        btic = time.time()
        # net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if args.log_interval and not (batch_iter + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}][iter {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_iter, batch_size / (time.time() - btic), name1, loss1, name2, loss2))
            btic = time.time()

            if ((batch_iter+1) % args.val_interval == 0) or (args.save_interval and (batch_iter+1) % args.save_interval == 0):
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][iter {}] Training cost: {:.3f} sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, batch_iter, (time.time() - tic), name1, loss1, name2, loss2))
                tic = time.time()
                ce_metric.reset()
                smoothl1_metric.reset()
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}][iter {}] Validation: \n{}'.format(epoch, batch_iter, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, best_map, current_map, batch_iter, args.save_interval, args.save_prefix)
            batch_iter += 1


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset_root)
    print(len(train_dataset))
    print(len(val_dataset))
    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name

    net = get_model('ssd_512_resnet50_v1_custom', classes=train_dataset.classes)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())

    for param in net.collect_params().values():
        # if param._data is not None:
        #     continue
        # print(param)
        param.initialize(force_reinit=True)

    train_data, val_data = get_dataloader(net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)

