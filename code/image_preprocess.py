# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:46:57 2018

@author: Srini
"""

from sklearn.externals.joblib import Parallel, delayed
import os
import numpy as np
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
ia.seed(1)
from datetime import datetime

###################################################

imagenamelist = os.listdir('../data/images_001/')

def image_preprocess(imagename):



    imagepath = '../data/images_001/' + imagename
    savepath = '../data/prep_images_001/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    img = cv2.resize(cv2.imread(imagepath), (224, 224)).astype(np.float32)

    np.save((savepath + imagename), img)

    return

tic = datetime.now()
Parallel(n_jobs=3,verbose = 1, backend= 'threading')(delayed(image_preprocess)(imagename) for imagename in imagenamelist)
print('time taken: {}'.format(datetime.now() - tic))

cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("preview", img)
cv2.waitKey(5000)
cv2.destroyWindow('preview')
