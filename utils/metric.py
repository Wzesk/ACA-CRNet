import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def compare_mae(img_true, img_test):
  img_true = img_true.astype(np.float32)
  img_test = img_test.astype(np.float32)
  return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def ssim(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    #error += compare_ssim(frames1[i], frames2[i], multichannel=True, win_size=11)
    error += compare_ssim(frames1[i], frames2[i], multichannel=True, win_size=51)
  return error/len(frames1)

def psnr(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    error += compare_psnr(frames1[i], frames2[i])
  return error/len(frames1)

def mae(frames1, frames2):
  error = 0
  for i in range(len(frames1)):
    error += compare_mae(frames1[i], frames2[i])

  return error/len(frames1)
