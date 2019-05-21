import os, cv2, csv
import numpy as np
from df_dataset import DF_Detection

# if __name__ == "__main__":
#     csv_file = open("../csv/train_label.csv")
#     lines = csv.reader(csv_file)
#     next(lines)
#     for line in lines:
#
#         bbox = [int(line[1]), int(line[2]), int(line[5]), int(line[6])]
#         if bbox[2] - bbox[0] > 200 or bbox[2] - bbox[0] < 18 or bbox[3] - bbox[1] < 18 or bbox[3] - bbox[1] > 200:
#             img_path = os.path.join("F:\DF/Train", line[0])
#             print(line)
#             if os.path.exists(img_path):
#                 cvimg = cv2.imread(img_path)
#                 cvimg = cv2.rectangle(cvimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
#                 cvimg = cv2.resize(cvimg, (1600, 900))
#                 # save_path = os.path.join("/media/handewei/新材料/DF/rect", line[0])
#                 cv2.imshow("dsfa", cvimg)
#                 # cv2.imwrite(save_path, cvimg)
#                 cv2.waitKey(0)

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


import os, cv2, csv
import numpy as np

def fix_line(line, local, left):
    new_line = ["img_name", '0', '0', '0', '0', '0', '0', '0', '0', '0']
    new_line[0] = local + line[0]
    new_line[1] = str(int(line[1]) - left)
    new_line[2] = str(int(line[2]) - 198)
    new_line[3] = str(int(line[3]) - left)
    new_line[4] = str(int(line[4]) - 198)
    new_line[5] = str(int(line[5]) - left)
    new_line[6] = str(int(line[6]) - 198)
    new_line[7] = str(int(line[7]) - left)
    new_line[8] = str(int(line[8]) - 198)
    new_line[9] = line[9]
    return new_line

root_rect = "F:/DF/rect"
root = "F:/DF/Train"
if __name__ == '__main__':
    csv_file = open("../csv/train_label.csv")
    csv_rect = open("../csv/rect.csv", "w", newline="")
    rect_writer = csv.writer(csv_rect)
    rect_writer.writerow(["filename", "X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "type"])
    lines = csv.reader(csv_file)
    next(lines)
    for line in lines:
        img_path = os.path.join(root, line[0])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if int(line[5]) < 1280:
                ROI = img[198: 1478, 0:1280]
                rect_path = os.path.join(root_rect, 'l' + line[0])
                cv2.imwrite(rect_path, ROI)
                new_line = fix_line(line, 'l', 0)
                rect_writer.writerow(new_line)

            if os.path.exists(img_path) and int(line[5]) < 2280 and int(line[1]) > 1000:
                ROI = img[198: 1478, 1000:2280]
                rect_path = os.path.join(root_rect, 'm' + line[0])
                cv2.imwrite(rect_path, ROI)
                new_line = fix_line(line, 'm', 1000)
                rect_writer.writerow(new_line)

            if os.path.exists(img_path) and int(line[1]) > 1920:
                ROI = img[198: 1478, 1920:3200]
                rect_path = os.path.join(root_rect, 'r' + line[0])
                cv2.imwrite(rect_path, ROI)
                new_line = fix_line(line, 'r', 1920)
                rect_writer.writerow(new_line)

        # cv2.imshow("img", ROI)
        # cv2.waitKey(1)
