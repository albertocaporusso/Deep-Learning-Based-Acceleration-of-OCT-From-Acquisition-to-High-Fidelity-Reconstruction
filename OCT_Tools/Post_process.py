import scipy
import numpy as np
from os import listdir
from os.path import isfile, join
import time

# LA MAGGIOR PARTE DELLE MODIFICHE SONO ALLA FUNZIONE ON_OCT_FRAME_READY (RIGA 645) + SAVE_FRAMES (RIGA 535)
#CARICO NPZ CON DATI NECESSARI

OCT_PROCT_AVERAGING = 30

mypath = "C:\\Users\\alber\\Desktop\\lavoro\\tesi\\dataset\\acquisizioni_conimaggini"

filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for fn in filenames:
    if fn[-4:] != '.npz':
        continue

    print('Loading file.. ' + fn)
    #LOADING NPZ FILE
    fn = mypath + "\\" + fn
    npfile = np.load(fn,allow_pickle=True)
    fn_no_dot = fn[0:-11] + fn[-10:-4]

    cur_spectra = npfile['cur_spectra'] #30840, 2048
    oct_ks = npfile['oct_ks'] #2048
    oct_spec_bg = npfile['oct_spec_bg'] #2048
    oct_even_ks = npfile['oct_even_ks'] #8192
    save_mode_z_counter = npfile['save_mode_z_counter']

    #START PROCESSING
    start = time.time()
    interp_func = scipy.interpolate.interp1d(oct_ks,(cur_spectra - oct_spec_bg),kind='cubic',axis=1) #interp_func(oct_even_ks) #shape 30840x8192

    print('Interpolation time: ' + str(time.time()-start)) #4/5 seconds if cubic, 0.5 if linear
    start = time.time()
    #cur_a_scan_array = abs(fftw.ifft2(interp_func(oct_even_ks),axes=(1,)))[:,0:np.round(np.size(oct_ks,0)/2).astype(int)]
    cur_a_scan_array = abs(np.fft.ifft2(interp_func(oct_even_ks),axes=(1,)))[:,0:np.round(np.size(oct_ks,0)/2).astype(int)] #shape 30840x1024
    print('IFFT time: ' + str(time.time()-start)) #10 to 12 seconds

    if (save_mode_z_counter % 2) == 0:
        oct_ascan_array = cur_a_scan_array.transpose()
    else:
        oct_ascan_array = (np.flipud(cur_a_scan_array)).transpose()

    oct_ascan_array = (np.flipud(cur_a_scan_array)).transpose() #1024,30840

    oct_mean_ascan_array = np.mean(oct_ascan_array.reshape(1024,-1, OCT_PROCT_AVERAGING), axis=2)#1024,1028 # average A-Scans

    np.save(fn_no_dot +".npy", oct_mean_ascan_array)

