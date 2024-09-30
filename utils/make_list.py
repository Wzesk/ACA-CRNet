import os

path="/home/hwl/hwl/datasets/remote-sense-data/SEN12MS-CR/dsen2cr-torch";

MusicNUM=0;
wc = 'png'
with open("/home/hwl/hwl/datasets/remote-sense-data/SEN12MS-CR/SEN12MS_CR_dsen2cr_torch_tif_list.txt",'a',encoding='utf-8') as filetext:
    for root,dirs,files in os.walk(path):
        for name in files:
            if name.endswith('.tif'):
                print(os.path.join(root,name));
                filetext.write(os.path.join(root,name)+"\n");
        # for name in dirs:
        #     print(os.path.join(root,name))
        #     filetext.write(os.path.join(root,name)+"\n");
    

filetext.close();
