import numpy as np
import cv2

# data paths
data_path = "./data_np/"
out_train_dir = "./data/images/"
train = "train.npy"

# load the diles
train_data = np.load(data_path+train)
nimages = train_data.shape[0]

print("loaded train data")

for i in range(0, nimages):
    # save each and every image
    a = train_data[i]
    # now make it as an bgr image
    img = cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_GRAY2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_train_dir+str(i)+".jpg", img)

# test_data = np.load(data_path+test)
# nimages = test_data.shape[0]

# for i in range(0, nimages):
#     # save each and every image
#     a = test_data[i]
#     # now make it as an bgr image
#     img = cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_GRAY2BGR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(out_test_dir+str(i)+".jpg", img)
