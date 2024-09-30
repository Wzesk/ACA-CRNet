# -*- coding: utf-8 -*-
"""
Created on May 9 10:24:49 2024

@author: Wenli Huang
"""
from models.ACA_CRNet import ACA_CRNet
import torch as t
from utils.utils import SaveImg,save_result_img
import os
import config.config_rice2 as config
from dataset.rice2_data_loader import dataloader as dataloader_r
import utils.np_metric as img_met
import utils.metrics_glf_cr as metrics_glf_cr
import numpy as np
import torch
import logging
import datetime
"""
steps:
1 define net, load model, dataloader
2 model inference
3 compute metrics: SSIM, PSNR
"""

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(level = logging.CRITICAL,
    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
    datefmt = '%Y-%m-%d(%a)%H:%M:%S',
    filename = 'test_ACA_CR_Rice2_log_'+now+'.txt',
    filemode = 'w')

#Log INFO level or higher to StreamHandler (default is standard error)
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s') #display real-time view, no time required
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
    
def predict(config):
    #define net
    gpu_ids = config.gpu_ids
    net = ACA_CRNet(config.in_ch,config.out_ch,config.alpha,config.num_layers,config.feature_sizes,gpu_ids)
    net = net.eval()

    dataloader = dataloader_r(config.predict_dataset_dir, isTrain=False, batch_size=1, shuffle=False,
                                nThreads=1, load_size=256, crop_size=128)
    logging.critical("test batch-epoch {} ".format(len(dataloader)))

    #net init
    if config.net_init is not None:
        param = t.load(config.net_init)
        net.load_state_dict(param)
        logging.critical("load params {}".format(config.net_init))
    else:
        logging.critical("Missing network path. Add after net_init in Config.py.")
        return

    #Load data to GPU if available and training with GPU.
    cloud_img=t.FloatTensor(config.batch_size,config.in_ch,config.width,config.height)
    ground_truth=t.FloatTensor(config.batch_size,config.out_ch,config.width,config.height)
    csm_img=t.FloatTensor(config.batch_size,1,config.width,config.height)

    #Put variables in GPU memory if using GPU.
    if config.use_gpu:
        net = net.cuda(gpu_ids[0])
        cloud_img=cloud_img.cuda(gpu_ids[0])
        ground_truth=ground_truth.cuda(gpu_ids[0])
        csm_img=csm_img.cuda(gpu_ids[0])

    MAE_vs = []
    MSE_vs = []
    RMSE_vs = []
    BRMSE_vs = []
    ssim_vs = []
    psnr_vs = []
    sam_vs = []
    logging.critical("start testing...")

    with t.no_grad():
        for iteration,batch in enumerate(dataloader,1):
            img_cld,img_csm,img_truth,patch_path_out=batch
            img_cld = img_cld * 2 - 1
            img_truth = img_truth * 2 - 1
            img_csm = img_csm *2 - 1
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)
            img_fake = net(img_cld)
            img_fake = t.tanh(img_fake)
            if not os.path.exists(config.output_dir + '/test_img'):
                os.makedirs(config.output_dir + '/test_img')

            outfilename = patch_path_out[0].split('/')[-1].split('.')[0]
            save_result_img(img_cld, os.path.join(config.output_dir + '/test_img', "{}_{}_incld.jpg".format(outfilename, iteration)))
            save_result_img(img_fake, os.path.join(config.output_dir + '/test_img', "{}_{}_outfake.jpg".format(outfilename, iteration)))
            save_result_img(img_truth, os.path.join(config.output_dir + '/test_img', "{}_{}_truth.jpg".format(outfilename, iteration)))
            save_result_img(img_csm, os.path.join(config.output_dir + '/test_img',
                                                    "{}_{}_mask.jpg".format(outfilename, iteration)))

            s2img1 = img_truth.clone()
            fake_img1 = img_fake.clone()

            # convert values from (-1)-1 to 0-1
            s2img1 = (s2img1 + 1) / 2.0
            fake_img1 = (fake_img1 + 1) / 2.0
            MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
            MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
            RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
            BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
            psnr_v = metrics_glf_cr.PSNR(s2img1 , fake_img1 )
            ssim_v = metrics_glf_cr.SSIM(s2img1 , fake_img1 )

            MAE_vs.append(np.asarray(MAE_v.cpu()))
            MSE_vs.append(np.asarray(MSE_v.cpu()))
            RMSE_vs.append(np.asarray(RMSE_v.cpu()))
            BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
            ssim_vs.append(np.asarray(ssim_v.cpu().detach().numpy()))
            psnr_vs.append(np.asarray(psnr_v))

            # spectral angle mapper
            mat = s2img1 * fake_img1
            mat = torch.sum(mat, 1)
            mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
            mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
            sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / np.pi)

            if not torch.isnan(sam_v):
                 sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))

            logging.critical(
                "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}".format(
                    MAE_v.cpu(), MSE_v.cpu(),
                    RMSE_v.cpu(), BRMSE_v.cpu(),
                    psnr_v, ssim_v.cpu(), sam_v))

        MAE_v = np.mean(MAE_vs)
        MSE_v = np.mean(MSE_vs)
        RMSE_v = np.mean(RMSE_vs)
        BRMSE_v = np.mean(BRMSE_vs)
        ssim_v = np.mean(ssim_vs)
        psnr_v = np.mean(psnr_vs)
        sam_v = np.mean(sam_vs)

        logging.critical(
            "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
                MAE_v, MSE_v, RMSE_v, BRMSE_v,
                psnr_v, ssim_v, sam_v))
        
if __name__=="__main__":
    myconfig=config.config()
    predict(myconfig)