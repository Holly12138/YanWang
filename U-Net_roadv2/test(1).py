import torch
import random
import math
from torch.autograd import Variable as V
from torchvision import transforms
import os
import numpy as np
import cv2
from time import time
from networks.UNet import UNet
from framework import MyFrame
#from loss import dice_bce_loss
from data_preprocessing import ImageFolder
#from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mask_to_submission import *
from time import time
from skimage import morphology

def img_float_to_uint8(img, PIXEL_DEPTH = 255):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def make_img_overlay(img, predicted_img, PIXEL_DEPTH = 255):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[ :, :, 0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def test():
    # test_transformer=transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor()])
    test_dir = "test_set_images/"
    n = len(os.listdir(test_dir))
    print("Loading " + str(n) + " test images")
    Results = []
    output_dir = "result/"
    test_imgs = np.asarray(
        [mpimg.imread(test_dir + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png') for i in range(n)])
    test_imgs = torch.tensor(test_imgs)
    for i in range(n):
        test_transformer=transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor()])
        print(str(i+1)+ "image")
        test_set = cv2.imread(test_dir + 'test_' + str(i + 1) + '/' + 'test_' + str(i + 1) + '.png')
        test = cv2.resize(test_set, (608,608))
        img = np.array(test, np.float32)/255.0 *3.2 -1.6
        ori_img = img
        img = V(torch.Tensor(img).to("cuda"))
        img = img.permute(2,0,1).unsqueeze(0)
        #print(img.shape)
        model = UNet()
        model.load_state_dict(torch.load('weights/UNet.pth'))
        model.to('cuda')
        model.eval()
        pred = model(img)
        pred = torch.squeeze(pred)
        #print(pred.shape)
        pred = pred.cpu()
        
        pred = torch.argmax(pred.permute(1,2,0),axis = -1).detach().numpy()
        #print(pred.shape)
        #print(pred)
        result = pred
        result[result<=0.25]=0
        result = morphology.remove_small_objects(result.astype(bool), 800)
        Results.append(result)
        #print(test_imgs[i,:,:,:].cpu().detach().numpy().shape)
        print(result.shape)
        #print(img[].cpu().detach().numpy().shape)
        
        new_img = make_img_overlay(ori_img,result)
        new_img.save(output_dir + str(i) + 'new_img.png')
    print(Results)
    
    #masks_to_submission(output_dir + "Unet.csv", Results)
    save_result(output_dir + "Unet.csv", Results)

if __name__ == "__main__":
   
    print("Test Phase")
    print("================================================")
    test()

    print("The Unet.csv is the final result for submission")
    