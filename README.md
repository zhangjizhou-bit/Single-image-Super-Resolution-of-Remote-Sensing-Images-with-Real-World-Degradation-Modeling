# Single-image-Super-Resolution-of-Remote-Sensing-Images-with-Real-World-Degradation-Modeling

![Graphic Abstract](https://www.mdpi.com/remotesensing/remotesensing-14-02895/article_deploy/html/images/remotesensing-14-02895-g001-550.jpg)

Paper: https://www.mdpi.com/2072-4292/14/12/2895

The code is based on natural image SR code by [Xiaozhong Ji et al](https://github.com/jixiaozhong/RealSR).


# Requirements
* numpy
* scipy
* pytorch
* torchvision
* lpips
* argparse
* yaml
* opencv-python

# Data preparation
* Prepare the AID dataset or other remote sensing image dataset.
* Use 'train.py' in './preprocess/KernelGAN/' to collect the kernel dataset. You may need to modify the path of input and output.
* Use 'collect_noise.py' in './preprocess/' to collect the noise patch dataset. You may need to modify the path of input and output in 'paths.yaml'.
* Generate the ideal or real-world training datasets with 'create_bicubic_dataset.py' or 'create_kernel_dataset.py' in './preprocess/'.

# Training
* Train models with ideal or real-world datasets with 'train.py' in the root path. You may need to modify the path in './options/aid/train_bicubic.yml' or './options/aid/train_kernel_noise.yml'.

# Test
* Train models with ideal or real-world datasets with 'test.py' in the root path. You may need to modify the path in './options/aid/test_aid.yml'.
