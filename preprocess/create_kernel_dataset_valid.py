import argparse
import os
import torch.utils.data
import yaml
import glob
import utils
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import sys
sys.path.append('d:/#11/codes/Real-SR-master/codes/')
from data.data_loader import noiseDataset
from KernelGAN.imresize import imresize
from scipy.io import loadmat
import numpy as np

parser = argparse.ArgumentParser(description='Apply the trained model to create a dataset')
parser.add_argument('--kernel_path', default='D:/Datasets/AID_all/kernels_train', type=str, help='kernel path to use')
parser.add_argument('--artifacts', default='clean', type=str, help='selecting different artifacts type')
parser.add_argument('--name', default='_real', type=str, help='additional string added to folder path')
parser.add_argument('--dataset', default='aid', type=str, help='selecting different datasets')
parser.add_argument('--track', default='valid', type=str, help='selecting train or valid track')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('./preprocess/paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)

if opt.dataset == 'df2k':
    path_sdsr = PATHS['datasets']['df2k'] + '/generated/sdsr/'
    path_tdsr = PATHS['datasets']['df2k'] + '/generated/tdsr/'
    input_source_dir = PATHS['df2k']['tdsr']['source']
    input_target_dir = PATHS['df2k']['tdsr']['target']
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = [os.path.join(input_target_dir, x) for x in os.listdir(input_target_dir) if utils.is_image_file(x)]
elif opt.dataset == 'dped':
    path_sdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_sdsr/'
    path_tdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_tdsr/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = []
elif opt.dataset == 'aid':
    path_sdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_sdsr/'
    path_tdsr = PATHS['datasets'][opt.dataset] + '/generated/' + opt.artifacts + '/' + opt.track + opt.name + '_tdsr/'
    input_source_dir = PATHS[opt.dataset][opt.artifacts]['hr'][opt.track]
    input_target_dir = None
    source_files = [os.path.join(input_source_dir, x) for x in os.listdir(input_source_dir) if utils.is_image_file(x)]
    target_files = []

tdsr_hr_dir = path_tdsr + 'HR'
tdsr_lr_dir = path_tdsr + 'LR'

# assert not os.path.exists(PATHS['datasets'][opt.dataset])


if not os.path.exists(tdsr_hr_dir):
    os.makedirs(tdsr_hr_dir)
if not os.path.exists(tdsr_lr_dir):
    os.makedirs(tdsr_lr_dir)

kernel_paths = glob.glob(os.path.join(opt.kernel_path, '*/*_kernel_x4.mat'))
kernel_num = len(kernel_paths)
print('kernel_num: ', kernel_num)

noises = noiseDataset('D:/Datasets/AID_all/noise_train/', 75)

# generate the noisy images
with torch.no_grad():
    for file in tqdm(source_files, desc='Generating images from source'):
        # load HR image
        input_img = Image.open(file)
        input_img = TF.to_tensor(input_img)

        # Resize HR image to clean it up and make sure it can be resized again
        resize2_img = utils.imresize(input_img, 1.0 / opt.cleanup_factor, True)
        _, w, h = resize2_img.size()
        w = w - w % opt.upscale_factor
        h = h - h % opt.upscale_factor
        resize2_cut_img = resize2_img[:, :w, :h]

        # Save resize2_cut_img as HR image for TDSR
        path = os.path.join(tdsr_hr_dir, os.path.basename(file))
        path = path[:-4] + '.png'
        resize2_cut_img = TF.to_pil_image(resize2_cut_img)
        resize2_cut_img.save(path, 'PNG')

        # Generate resize3_cut_img and apply model
        kernel_path = kernel_paths[np.random.randint(0, kernel_num)]
        mat = loadmat(kernel_path)
        k = np.array([mat['Kernel']]).squeeze()
        resize3_cut_img = imresize(np.array(resize2_cut_img), scale_factor=1.0 / opt.upscale_factor, kernel=k)

        # Save resize3_cut_img as LR image for TDSR
        path = os.path.join(tdsr_lr_dir, os.path.basename(file))
        path = path[:-4] + '.png'
        noise = noises[np.random.randint(0, len(noises))]
        img = TF.to_tensor(resize3_cut_img)
        img = np.clip(img + noise, 0, 1)
        TF.to_pil_image(img).save(path, 'PNG')

