# -*- coding: utf-8 -*-
"""

@author: Muhammad Dawood
"""
# importing necessary libraries
from keras.models import load_model
from keras.preprocessing import image
import netCDF4 as nc4
import scipy
import numpy as np

# Loading keras model

model = load_model('./model/model.h5')
def load_image(filename):
    if filename.endswith(".nc"):
        f = nc4.Dataset(filename,'r')
        im=f.variables['IRWIN']
        img=np.array(im)
        img=np.reshape(img,(301,301))
        result=scipy.misc.imresize(img,(224,224))
        x = image.img_to_array(result)
        x = np.expand_dims(x, axis=0)
    return x

processed_image = load_image('test3.nc')
print('Models predicted intensities', model.predict(processed_image)[0][0])



