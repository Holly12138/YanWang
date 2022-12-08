import torch
import random
import math


import os
import numpy as np

from time import time
from networks.UNet import UNet
from framework import MyFrame
from data_preprocessing import ImageFolder
from tqdm import tqdm
# SEED = 0
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
def train():
    # The network need the size to be a multiple of 32, resize is intriduced
    model_type = 'UNet'


    # Loading the name of training images and groundtruth images
    train_root = 'training/'
    image_root = os.path.join(train_root, 'images')
    gt_root = os.path.join(train_root, 'groundtruth')
    image_list = np.array(sorted(
        [f for f in os.listdir(image_root) if f.endswith('.png')]))
    # Randomly select 20% of training data for validation
    total_data_num = image_list.shape[0]
    validation_data_num = math.ceil(total_data_num * 0.1)
    validation_idx = random.sample(range(total_data_num), validation_data_num)
    new_train_indx = list(
        set(range(total_data_num)).difference(set(validation_idx)))

    val_img_list = image_list[validation_idx].tolist()
    image_list = image_list[new_train_indx].tolist()

    solver = MyFrame(UNet,  2e-4)
    train_batchsize = 6
    val_batchsize = 6



    # Data preprocessing for training set
    train_dataset = ImageFolder(image_list, image_root, gt_root, (384, 384))
    # No data preprocessing for validation dataset
    val_dataset = ImageFolder(val_img_list, image_root, gt_root,(384, 384), False)

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True,
        num_workers=0)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batchsize,
        shuffle=True,
        num_workers=0)

    if not os.path.exists('logs/'):
        os.mkdir('logs/')

    mylog = open('logs/'+model_type+'.log', 'w')
    tic = time()
    no_optim = 0
    no_optim_valid = 0
    total_epoch = 500
    train_epoch_best_loss = 100.
    validation_epoch_best_loss = 100

    for epoch in range(1, total_epoch + 1):
        print('---------- Epoch:'+str(epoch) + ' ----------')
        data_loader_iter = tqdm(data_loader)
        train_epoch_loss = 0
        validation_epoch_loss = 0
        train_epoch_dice=0
        validation_epoch_dice=0

        print('Train:')
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss,train_dice = solver.optimize()
            train_epoch_loss += train_loss
            train_epoch_dice += train_dice
        train_epoch_loss /= len(data_loader_iter)
        train_epoch_dice /= len(data_loader_iter)

        # Writing log
        duration_of_epoch = int(time()-tic)
        mylog.write('********************' + '\n')
        mylog.write('--epoch:' + str(epoch) + '  --time:' + str(duration_of_epoch) + '  --train_loss:' + str(
            train_epoch_loss)+'--train_dice'+str(train_epoch_dice) + '\n')
        # Print training loss
        print('--epoch:', epoch, '  --time:', duration_of_epoch, '  --train_loss:',
              train_epoch_loss,'--train_dice:',train_epoch_dice)

        #  Do validation every 5 epochs
        if epoch % 5 == 0:
            val_data_loader_iter = tqdm(val_data_loader)

            print("Validation: ")
            for val_img, val_mask in val_data_loader_iter:
                solver.set_input(val_img, val_mask)
                val_loss,val_dice = solver.optimize(True)
                validation_epoch_loss += val_loss
                validation_epoch_dice += val_dice
            validation_epoch_loss /= len(val_data_loader_iter)
            validation_epoch_dice /= len(val_data_loader_iter)
            # Writing log
            mylog.write('--epoch:' + str(epoch) +
                        '  --validation_loss:' + str(validation_epoch_loss) +'--validation_dice:'+str(validation_epoch_dice) +'\n')
            # Print validation loss
            print('--epoch:', epoch,  '  --validation_loss:',
                  validation_epoch_loss,'--validation_dice:',validation_epoch_loss)
            if validation_epoch_loss < validation_epoch_best_loss:
                no_optim_valid = 0
                validation_epoch_best_loss = validation_epoch_loss
                # Store the weight
                solver.save('weights/'+model_type+'.pth')
            else:
                no_optim_valid += 1
                if no_optim_valid >= 3:
                    # Early Stop
                    mylog.write(
                        'Validation loss not improving, early stop at' + str(epoch)+'epoch')
                    print(
                        'Validation loss not improving, early stop at %d epoch' % epoch)
                    break

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss

        if no_optim > 5:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/' + model_type+ '.pth')
            #  Adjust learning reate
            solver.update_lr(1.1, factor=True, mylog=mylog)
        mylog.flush()

    mylog.write('Finish!')
    print('train end')
    mylog.close()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    train()