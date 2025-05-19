# -*- coding: utf-8 -*-
"""
Created on May 9 10:24:49 2024

@author: Wenli Huang
"""
import os
from utils.utils import SaveImg,save_result_img
import torch as t
from models.ACA_CRNet import ACA_CRNet
import utils.visualize as visualize
import utils.utils as utility
import time
import utils.img_operations as imgop
#import config.config_rice1 as config
#import config.config_rice1 as config
import config.config_rice1_test as config
import utils.np_metric as img_met
import numpy as np
from dataset.rice1_data_loader import dataloader
import logging
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(level = logging.CRITICAL,
    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
    datefmt = '%Y-%m-%d(%a)%H:%M:%S',
    filename = 'train_ACA_CR_Rice1_log_'+now+'.txt',
    filemode = 'w')

#Log INFO level or higher to StreamHandler (default is standard error)
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s') #display real-time view, no time required
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def train(config):
    #Notice using GPU
    if config.use_gpu==False and t.cuda.is_available()==True:
        logging.critical("NVIDIA GPU {} available, set use_gpu to True in config.py.".format(t.cuda.get_device_name()))
    else:
        logging.critical("NVIDIA GPU {} for Training".format(t.cuda.get_device_name()))

    #Visualize Operator
    vis = visualize.Visualizer("ACA_CRNet")

    #dataset and dataloader
    logging.critical("Creating dataset and dataloader...")
    
    train_dataloader = dataloader(config.train_datset_dir,isTrain=True,batch_size=config.batch_size,shuffle=True,nThreads=config.batch_size,load_size=256,crop_size=128)
    val_dataloader = dataloader(config.val_dataset_dir, isTrain=False, batch_size=1, shuffle=True,
                               nThreads=1, load_size=256, crop_size=128)

    val_dataiter=iter(val_dataloader)
    logging.critical("Train dataset len：{}\r\nValid dataset len：{}\r\ndataset init done".format(len(train_dataloader), len(val_dataloader)))

    #define net
    gpu_ids = config.gpu_ids
    net = ACA_CRNet(config.in_ch,config.out_ch,config.alpha,config.num_layers,config.feature_sizes,gpu_ids)

    metrics = [img_met.cloud_mean_absolute_error,
               img_met.cloud_mean_squared_error,
               img_met.cloud_psnr,
               img_met.cloud_root_mean_squared_error, img_met.cloud_bandwise_root_mean_squared_error,
               img_met.cloud_ssim]

    #如果有初始化的网络路径 则初始化
    if config.net_init is not None:
        param = t.load(config.net_init)
        net.load_state_dict(param)
        logging.critical("load params {}".format(config.net_init))
    
    opt = t.optim.Adam(net.parameters(),lr=config.lr,betas=(config.beta1,0.999))
    
    #loss function
    CARL_Loss = imgop.carl_error_l1
    
    #Load data to GPU if available and training with GPU.
    cloud_img=t.FloatTensor(config.batch_size,config.in_ch,config.width,config.height)
    ground_truth=t.FloatTensor(config.batch_size,config.out_ch,config.width,config.height)
    # csm_img=t.FloatTensor(config.batch_size,1,config.width,config.height)
    
    #Put variables in GPU memory if using GPU.
    if config.use_gpu:
        net = net.cuda(gpu_ids[0])
        cloud_img=cloud_img.cuda(gpu_ids[0])
        ground_truth=ground_truth.cuda(gpu_ids[0])
        # csm_img=csm_img.cuda(gpu_ids[0])
    
    logging.critical("start traing...")

    for epoch in range(1,config.epoch):
        epoch_start_time = time.time()
        for iteration,batch in enumerate(train_dataloader):
            #data operate, convert numpy data to tensor
            img_cld,img_truth,patch_path_out=batch
            img_cld = img_cld  * 2 - 1
            img_truth = img_truth * 2 - 1
            
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            #img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)

            img_fake = net(img_cld)

            img_fake = t.tanh(img_fake)
            opt.zero_grad()
            loss= CARL_Loss(img_truth,img_fake)
            loss.backward() 
            opt.step()

            MAE_vs =[]
            MSE_vs = []
            RMSE_vs = []
            BRMSE_vs = []
            ssim_vs = []
            psnr_vs = []
            if False:
                #Visualize loss, and save img
                if iteration % config.show_freq == 0:
                     with t.no_grad():
                         print("epoch[{}]({}/{}):loss_fake:{:.8f}".format(
                              epoch,iteration,len(train_dataloader),loss.item()))
                         logging.critical("epoch[{}]({}/{}):loss_fake:{:.8f}".format(
                              epoch,iteration,len(train_dataloader),loss.item()))
                         #valid step
                         inputdata,s2img,patch_path_out=next(val_dataiter)
                         inputdata1 = inputdata * 2 - 1
                         s2img1 = s2img * 2 - 1
                         inputdata_cuda=inputdata1.cuda()
                         net.eval()
                         fake_img = net(inputdata_cuda)
                         fake_img = t.tanh(fake_img)
                         net.train()

                         if not os.path.exists(config.output_dir+'/val_img'):
                             os.makedirs(config.output_dir+'/val_img')
                         save_result_img(inputdata1, os.path.join(config.output_dir+'/val_img', "{}_cloud_iteration.jpg".format(iteration)))
                         save_result_img(s2img1, os.path.join(config.output_dir+'/val_img', "{}_truth_iteration_.jpg".format(iteration)))
                         save_result_img(fake_img, os.path.join(config.output_dir+'/val_img', "{}_fake_iteration.jpg".format(iteration)))
                         vis.plot("loss_d_fake", loss.item())
                         s2img1 = s2img.clone()
                         fake_img1 = fake_img.clone()
                         fake_img1 = (fake_img1+1)/2.0
                         s2img1 = s2img1.cuda()
                         MAE_v = img_met.cloud_mean_absolute_error(s2img1,fake_img1)
                         MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
                         RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
                         BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
                         #ssim_v = img_met.cloud_ssim(s2img1, fake_img1)
                         psnr_v = img_met.cloud_psnr(s2img1, fake_img1)
                         MAE_vs.append(np.asarray(MAE_v.cpu()))
                         MSE_vs.append(np.asarray(MSE_v.cpu()))
                         RMSE_vs.append(np.asarray(RMSE_v.cpu()))
                         BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
                         #ssim_vs.append(np.asarray(ssim_v))
                         psnr_vs.append(np.asarray(psnr_v.cpu()))

                     MAE_v = np.mean(MAE_vs)
                     MSE_v = np.mean(MSE_vs)
                     RMSE_v = np.mean(RMSE_vs)
                     BRMSE_v = np.mean(BRMSE_vs)
                     #ssim_v = np.mean(ssim_vs)
                     psnr_v = np.mean(psnr_vs)
                     logging.critical("MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f}".format(MAE_v, MSE_v, RMSE_v, BRMSE_v,
                                                                                         psnr_v))
            #save model
            if iteration%config.save_frequency==0:
                utility.save_state_dict(net, epoch,iteration,config.net_state_dict_save_dir)
                logging.critical(
                    "save state epoch {} iter {}".format(epoch, iteration))

        utility.save_state_dict(net, epoch, iteration, config.net_state_dict_save_dir)
        logging.critical(
            "save state epoch {} iter {}".format(epoch, iteration))
        logging.critical(
            "complete epoch {:.6f}, use time {:.6f}s".format(epoch,time.time()-epoch_start_time))

if __name__=="__main__":
    myconfig=config.config()
    train(myconfig)
