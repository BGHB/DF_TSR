import os
import numpy as np
import random
import cv2, csv
import mxnet as mx
from gluoncv.data.base import VisionDataset

class DF_Detection(VisionDataset):
    CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
               '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')

    def __init__(self, root, label_name, transform=None, item=None):
        super(DF_Detection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._csv_path = os.path.join(self._root, label_name)
        self._image_root = os.path.join(self._root, 'Train')
        self._transform = transform
        self.index_map = dict(zip(self.classes, range(self.num_class)))
        self._items = item or self._load_items()
        self.labels = []

    def __str__(self):
        return self.__class__.__name__

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_name = self._items[idx][0]

        label = [[int(self._items[idx][1]),  # xmin
                 int(self._items[idx][2]),  # ymin
                 int(self._items[idx][5]),  # xmax
                 int(self._items[idx][6]),  # ymax
                 int(self._items[idx][9])   # id
        ]]
        img_path = os.path.join(self._image_root, img_name)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, np.array(label)

    def _load_items(self):
        """Parse csv file and return labels."""
        items = []
        csv_file = open(self._csv_path)
        lines = csv.reader(csv_file)
        next(lines)
        for line in lines:
            items.append(line)
        return items

    # def split(self, train_ratio, validate_ratio, test_ratio, shuffle=False):
    #     num_pre = len(self._items) / (train_ratio + validate_ratio + test_ratio)
    #     import math
    #     train_num = math.floor(num_pre * train_ratio)
    #     validate_num = math.floor(num_pre * validate_ratio)
    #     if shuffle:
    #         import random
    #         random.seed(233)
    #         random.shuffle(self._items)
    #     train_items = self._items[:train_num]
    #     validate_items = self._items[train_num:train_num + validate_num]
    #     test_items = self._items[train_num + validate_num:]
    #     return DF_Detection(self._root, self._transform, train_items), \
    #            DF_Detection(self._root, self._transform, validate_items), \
    #            DF_Detection(self._root, self._transform, test_items)


if __name__ == "__main__":
    root = "/media/handewei/新材料/DF"

    train_dataset = DF_Detection(root, label_name='train_label.csv')
    val_dataset = DF_Detection(root, label_name='val_label.csv')
    print(len(train_dataset), len(val_dataset))
    # train_data, val_data, _ = dataset.split(1, 0, 0)
    # print(len(train_data), len(val_data))

    for train_image, train_label in train_dataset:
        bbox = train_label[:, :4]
        cid = train_label[:, 4:5]
        cvimg = train_image.asnumpy()

        cvimg = cv2.rectangle(cvimg, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), (0, 255, 0), 8)
        cvimg = cv2.putText(cvimg, str(cid[0][0]), (bbox[0][2], bbox[0][3]), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 8)
        # ROI = cvimg[bbox[1]:bbox[3], bbox[0]: bbox[2], :]
        cvimg = cv2.resize(cvimg, (cvimg.shape[1]//4, cvimg.shape[0]//4))

        cv2.imshow("dsfa", cvimg)
        cv2.waitKey(1)
        # rect_name = str(cid) + "-" + str(index) + ".jpg"
        # save_path = os.path.join("F:/DF/rect", rect_name)
        # print(save_path)
        # cv2.imwrite(save_path, ROI)