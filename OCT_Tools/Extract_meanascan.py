from os import listdir
from os.path import isfile, join
import numpy as np

mypath = r"C:\\Users\\alber\\Desktop\\lavoro\\tesi\\dataset\\acquisizioni_conimaggini"
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for fn in filenames:
    if fn[-4:] != '.npz':
        continue

    print('Loading file.. ' + fn)

    fn = mypath + "\\" + fn
    npfile = np.load(fn,allow_pickle=True)

    masa = npfile['mean_a_scan_array'] #1024x1028

    fn_no_dot = fn[0:-11] + fn[-10:-4] # skip "." for Slicer
    np.save(fn_no_dot + ".npy",masa)