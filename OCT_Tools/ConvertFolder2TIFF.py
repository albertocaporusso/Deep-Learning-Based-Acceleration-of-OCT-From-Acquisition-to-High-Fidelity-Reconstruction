from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.ndimage import geometric_transform
from PIL import Image
import time

def to_polar(img, order=5):
    num_angles = img.shape[1] #second dimension of mean a scan array
    def transform(coords):
        xindex,yindex = coords
        #♣x0, y0 = (200,512)
        x0, y0 = (img.shape[0], img.shape[0])  #first dimension of mean a scan array, 1024
        x = xindex - x0
        y = yindex - y0
        r = np.sqrt(x ** 2.0 + y ** 2.0)*( img.shape[1]/num_angles)
        theta = np.arctan2(y, x,where=True)
        theta_index = (theta + np.pi) * img.shape[1] / (2 * np.pi)
        return (r,theta_index)
    polar = geometric_transform(img, transform, order=order,mode='constant',prefilter=True, output_shape=(2048,2048))
    return polar

mypath = "C:\\Users\\alber\\Desktop\\lavoro4r\\lavoroUniVR\\tesi\\dataset\\2025-02-26_12-35-13_854538_fat3"

filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for fn in filenames:
    if fn[-4:] != '.npy':
        continue

    print('Loading file.. ' + fn)

    fn = mypath + "\\" + fn
    npfile = np.load(fn,allow_pickle=True)

    res_log = np.log(npfile) #1024,1028
    start = time.time()
    res = to_polar(res_log) #2048,2048
    print('Image Generation Time: ' + str(time.time() - start))
    image = Image.fromarray(res)
    image.save(fn[:-4] + ".tiff")

