import os, cv2, csv
import numpy as np
from df_dataset import DF_Detection

if __name__ == "__main__":
    csv_file = open("/media/handewei/新材料/DF/val_label.csv")
    lines = csv.reader(csv_file)
    next(lines)
    for line in lines:
        bbox = [int(line[1]), int(line[2]), int(line[5]), int(line[6])]
        if bbox[2] - bbox[0] > 200 or bbox[2] - bbox[0] < 18 or bbox[3] - bbox[1] < 18 or bbox[3] - bbox[1] > 200:
            img_path = os.path.join("/media/handewei/新材料/DF/Train", line[0])
            print(line[0])
            if os.path.exists(img_path):
                cvimg = cv2.imread(img_path)
                cvimg = cv2.rectangle(cvimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
                cvimg = cv2.resize(cvimg, (1600, 900))
                save_path = os.path.join("/media/handewei/新材料/DF/rect", line[0])
                cv2.imshow("dsfa", cvimg)
                cv2.imwrite(save_path, cvimg)
                cv2.waitKey(1)


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
