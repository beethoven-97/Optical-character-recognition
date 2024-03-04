import cv2
import numpy as np
from preprocess import Preprocess, train, valid, mem, chars
from model import Model1
from tensorflow.python.keras import backend


def test(img):
    # resize
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)

    # normalize
    img = img / 255.
    print(np.array(img).shape)


    # test
    m = Model1()
    m1 = m.crnn()
    m1.load_weights(m.best_model)
    preds = m1.predict(np.array(img))
    decoder = backend.get_value(
        backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])

    # result
    for i, x in enumerate(decoder):
        print('pred=', end='')
        for j in x:
            if int(j) != -1:
                print(chars[int(j)], end='')
        print('\n')


# import image
image = cv2.imread('input.jpg')
# cv2.imshow('orig',image)
# cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('second', thresh)
cv2.waitKey(0)

# dilation
kernel = np.ones((5, 100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

# find contours
im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = gray[y:y + h, x:x + w]

    # show ROI
    # print(x, y, w, h)
    cv2.imshow('segment no:' + str(i), roi)
    #test(roi)
    cv2.imwrite("segment_no_" + str(i) + ".png", roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
    cv2.waitKey(0)

# cv2.imwrite('final_bounded_box_image.png', image)
# cv2.imshow('marked areas', image)
# cv2.waitKey(0)
