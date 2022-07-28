import cv2
import glob
import numpy as np
import os.path as osp
import torch
import lpips
from torchvision.transforms.functional import normalize
from scipy.io import savemat

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def img2tensor(img):
    img = img.transpose(2, 0, 1).astype('float32')/255
    tensor = torch.from_numpy(img)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    tensor_norm = normalize(tensor, mean, std).cuda()
    return tensor_norm


def main():
    data_name = 'AID'
    data_train = 'bicubic'
    data_test = 'bicubic'


    folder_hr = 'D:/Datasets/AID_all/generated/clean/valid_real_tdsr/HR'
    folders_sr = [  'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/'+data_test+'_upsample'+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/srcnn_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/vdsr_'+data_train+'_'+data_test+'/'+data_name,
                    # 'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/lapsrn_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/ddbpn_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/edsr_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/srgan_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/drln_'+data_train+'_'+data_test+'/'+data_name,
                    'D:/#11/codes/Real-SR-master/codes/results_'+data_name+'/rban_unet_'+data_train+'_'+data_test+'/'+data_name]

    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    img_hr_list = sorted(glob.glob(osp.join(folder_hr, '*')))

    num_methods = len(folders_sr)
    num_imgs = len(img_hr_list)

    lpips_all = np.zeros((num_methods, num_imgs))
    for idx_img in range(num_imgs):
    # for idx_img in range(10):
        if ( (idx_img+1) % 10 == 0):
            print(idx_img+1)
        path_hr = img_hr_list[idx_img]
        img_hr = cv2.cvtColor(cv2.imread(path_hr, cv2.IMREAD_UNCHANGED) ,cv2.COLOR_BGR2RGB)
        tensor_norm_hr = img2tensor(img_hr)
        file_name = osp.basename(path_hr)

        for idx_method in range(num_methods):
            path_sr = osp.join(folders_sr[idx_method], file_name)
            img_sr = cv2.cvtColor(cv2.imread(path_sr, cv2.IMREAD_UNCHANGED) ,cv2.COLOR_BGR2RGB)
            tensor_norm_sr = img2tensor(img_sr)
            
            lpips_all[idx_method, idx_img] = loss_fn_alex(tensor_norm_hr, tensor_norm_sr).detach().cpu().numpy()

    lpips_mean = np.mean(lpips_all, axis=1)

    mat_name = 'D:/#11/codes/Real-SR-master/codes/matlab_metrics/lpips_'+data_name+'_'+data_train+'_'+data_test+'.mat'
    savemat(mat_name, {'lpips':lpips_mean})



def main_ablation():

    folder_hr = 'D:/Datasets/AID_all/generated/clean/valid_real_tdsr/HR'
    folders_sr = [  'D:/#11/codes/Real-SR-master/codes/results_ablation/real_upsample/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_unet_bicubic_real/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_unet_noblur/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_unet_nonoise/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rgn_real_real/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_real_real/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_vgg_real_real/AID/',
                    'D:/#11/codes/Real-SR-master/codes/results_ablation/rban_unet_real_real/AID/']

    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    img_hr_list = sorted(glob.glob(osp.join(folder_hr, '*')))

    num_methods = len(folders_sr)
    num_imgs = len(img_hr_list)

    lpips_all = np.zeros((num_methods, num_imgs))
    for idx_img in range(num_imgs):
    # for idx_img in range(10):
        if ( (idx_img+1) % 10 == 0):
            print(idx_img+1)
        path_hr = img_hr_list[idx_img]
        img_hr = cv2.cvtColor(cv2.imread(path_hr, cv2.IMREAD_UNCHANGED) ,cv2.COLOR_BGR2RGB)
        tensor_norm_hr = img2tensor(img_hr)
        file_name = osp.basename(path_hr)

        for idx_method in range(num_methods):
            path_sr = osp.join(folders_sr[idx_method], file_name)
            img_sr = cv2.cvtColor(cv2.imread(path_sr, cv2.IMREAD_UNCHANGED) ,cv2.COLOR_BGR2RGB)
            tensor_norm_sr = img2tensor(img_sr)
            
            lpips_all[idx_method, idx_img] = loss_fn_alex(tensor_norm_hr, tensor_norm_sr).detach().cpu().numpy()

    lpips_mean = np.mean(lpips_all, axis=1)

    mat_name = 'D:/#11/codes/Real-SR-master/codes/matlab_ablation_metrics/lpips_ablation.mat'
    savemat(mat_name, {'lpips':lpips_mean})

if __name__ == '__main__':
    main()
    # main_ablation()