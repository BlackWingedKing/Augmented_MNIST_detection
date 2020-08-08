import numpy as np
import cv2
import pickle
from PIL import Image

data_dir = "./data/"
train_data = data_dir+"train.npy"
train_labels = data_dir + "train_label.pkl"

# read the images
a = np.load(train_data)
with open(train_labels, "rb") as fp:
    labels = pickle.load(fp)

print("loaded images and labels")

tot = min(100, a.shape[0])

pim_list = []
for i in range(0,tot):
    color_image = cv2.cvtColor(a[i].astype(np.uint8),cv2.COLOR_GRAY2BGR)
    # draw bounding boxes over the images accordingly
    targets = labels[i][1]
    for j in targets:
        cv2.rectangle(color_image, (j[1], j[0]), (j[1]+j[2], j[0]+j[2]), (0, 255, 0), 2) 
    pimg = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
    pimg = Image.fromarray(pimg)
    pim_list.append(pimg)

pim_list[0].save('visdata.gif',
               save_all=True, append_images=pim_list[1:], optimize=False, duration=100, loop=0)
