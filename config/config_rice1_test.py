# -*- coding: utf-8 -*-
"""
Created on May 9 10:24:49 2024

@author: Wenli Huang
"""


class config():
    #datasets config
    train_datset_dir =   "./coastal-datasets/RICE1/train/cloud/" # The path of the image to be train
    predict_dataset_dir ="./coastal-datasets/RICE1/test/cloud/"  # The path of the image to be predicted in testing
    val_dataset_dir =    "./coastal-datasets/RICE1/test/cloud/"  # The path of the image to be valid in training

    width= 128#image size 128
    height= 128
    threads= 0 #num_of_workers, In the Windows environment, values greater than 0 may lead to a bug, known as a broken pipe

    output_dir = "./results" # output img dir
    net_state_dict_save_dir = './experiments/rice1-models/' # output pth dir

    #train options
    use_gpu = True
    save_frequency = 5000 #How many iterations to save a pth file?
    
    show_freq = 100 #How many iters of testing and results presentation
    batch_size = 2 # batchsize train bs=12
    # net init pth, train net_init=None
    net_init = "./experiments/rice1-models/net_epoch_2_iteration_199.pth" # None
    gpu_ids=[0] #GPU ID
    epoch = 3 #300 # train 300
    lr= 7e-5
    beta1= 0.9#ADAM optimize beta1
    in_ch= 3#input channel
    out_ch = 3#output channel
    alpha =0.1#resnet aplha
    num_layers=16#resnet layers
    feature_sizes=256#resnet feature