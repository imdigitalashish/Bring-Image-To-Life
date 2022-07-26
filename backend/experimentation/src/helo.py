import numpy as np
import argparse
import cv2 as cv


prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = "models/pts_in_hull.npy"
image_path = "ansel_adams.jpg"
W_in = 224
H_in = 224
imshowSize = (640, 480)

# args = parse_args()

# Select desired model
net = cv.dnn.readNetFromCaffe(prototxt_path, model_path)

pts_in_hull = np.load(kernel_path) # load cluster centers

# populate cluster centers as 1x1 convolution kernel
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

cap = cv.VideoCapture(0)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)

    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    img_l = img_lab[:,:,0] # pull out L channel
    (H_orig,W_orig) = img_rgb.shape[:2] # original image size

    # resize image to network input size
    img_rs = cv.resize(img_rgb, (W_in, H_in)) # resize image to network input size
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:,:,0]
    img_l_rs -= 50 # subtract 50 for mean-centering

    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result

    (H_out,W_out) = ab_dec.shape[:2]
    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    frame = cv.resize(frame, imshowSize)
    cv.imshow('origin', frame)
    cv.imshow('gray', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
    cv.imshow('colorized', cv.resize(img_bgr_out, imshowSize))