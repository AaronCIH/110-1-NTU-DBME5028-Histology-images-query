import argparse
import torch
from dataset import Histology_testdata
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from torchvision import models
from models.vae import VanillaVAE
from functools import wraps
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os
import tensorboardX as tbx
import torchvision as tv
from evualate import euclidean_dist

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def get_feature(model, img):
    for index, layer in enumerate(model.children()):
        if index <= 8:
            img = layer(img).squeeze()
    return img.squeeze()

def vis_sample(input, vae, residual, enhance, save_dir=''):
    img_w, img_h, img_dims = input.shape
    plt.figure(figsize=(20, 20))
    imgs = [input, vae, residual, enhance]
    ####### 處理 img #######
    full_img = np.hstack(imgs)
    ####### plot 結果 #########
    plt.axis('off')
    fig = plt.imshow(full_img)
    ####### 處理 labels ########
    encoder = ['input', 'vae', 'residual', 'enhance']
    for ct, code in enumerate(encoder):
       plt.text(ct*img_w+ct*14, -5, "%s"%code, color="blue", fontsize=24, fontweight='bold')
    plt.savefig(save_dir)
    plt.show()      

# rank list visualize
def read_im(im_path):
  im = np.asarray(Image.open(im_path))  # shape [H, W, 3]
  resize_h_w = (384, 292)    # Resize to (im_h, im_w) = (256, 256)
  if (im.shape[0], im.shape[1]) != resize_h_w:
    im = cv2.resize(im, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
  im = im.transpose(2, 0, 1)    # shape [3, H, W]
  return im

def make_im_grid(ims, n_rows, n_cols, space, pad_val):
    """Make a grid of images with space in between.
    Args:
        ims: a list of [3, im_h, im_w] images
        n_rows: num of rows
        n_cols: num of columns
        space: the num of pixels between two images
        pad_val: scalar, or numpy array with shape [3]; the color of the space
    Returns:
        ret_im: a numpy array with shape [3, H, W]
    """
    assert (ims[0].ndim == 3) and (ims[0].shape[0] == 3)
    assert len(ims) <= n_rows * n_cols
    h, w = ims[0].shape[1:]
    H = h * n_rows + space * (n_rows - 1)
    W = w * n_cols + space * (n_cols - 1)
    if isinstance(pad_val, np.ndarray):
        # reshape to [3, 1, 1]
        pad_val = pad_val.flatten()[:, np.newaxis, np.newaxis]
    ret_im = (np.ones([3, H, W]) * pad_val).astype(ims[0].dtype)
    for n, im in enumerate(ims):
        r = n // n_cols
        c = n % n_cols
        h1 = r * (h + space)
        h2 = r * (h + space) + h
        w1 = c * (w + space)
        w2 = c * (w + space) + w
        ret_im[:, h1:h2, w1:w2] = im
    return ret_im    

def rank_list_to_im(rank_list, q_im_path, g_im_paths, n_row=-1, n_col=-1, save_dir=''):
    '''
    save a query and its rank list as an image.
    Args:
        rank_list: a list, the indices of gallery images to show
        q_im_path: query image path
        g_im_paths: ALL gallery image paths
    '''
    ims = [read_im(q_im_path)]
    for idx, ind in enumerate(rank_list):
        im = read_im(g_im_paths[ind])
        ims.append(im)
    # set row, col
    if n_row == -1 and n_col == -1:
        n_row, n_col = 1, len(rank_list) + 1
    assert n_row * n_col == len(rank_list) + 1
    # make grid
    im = make_im_grid(ims, n_row, n_col, 8 ,255)
    result = Image.fromarray(im.transpose(1,2,0))
    result = result.save(save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--load_model", type=str, default='/work/r09921058/DeepLearning/DBME/Final/CIH_code2/VAE_BYOL/checkpoints/resnext101/model_final_threshold_0.53_acc77.42.pth', help="path to the model of generator")
    parser.add_argument("--load_attention", type=str, default='/work/r09921058/DeepLearning/DBME/Final/CIH_code2/VAE_BYOL/checkpoints/resnext101/attention_final_threshold_0.53_acc77.42.pth', help="path to the model of generator")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--threshold", type=float, default=0.51, help="threshold for classification")
    parser.add_argument("--image_size", type=float, default=224, help="size of image")
    """base_options"""
    parser.add_argument("--data", type=str, default='/work/r09921058/DeepLearning/DBME/Final/Dataset', help="path to dataset")
    parser.add_argument("--output_dir", type=str, default='./result/exper_vis/', help="name of saved model name")
    parser.add_argument("--csv_name", type=str, default='prediction.csv', help="name of saved model name")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument('--lamda', default=0.001, type=float, help='Lmse + \lamda * Lkl')
    parser.add_argument('-v', '--vis', action='store_true', default=False, help='Model vae module')
    opt = parser.parse_args()

    test_data = Histology_testdata(opt.data, opt.image_size)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=opt.n_cpu)

    os.makedirs(opt.output_dir, exist_ok=True)

    # attention
    attention = VanillaVAE(in_channels=3,
                        latent_dim=512,
                        hidden_dims=None,
                        W_Lkld=opt.lamda).cuda()
    attention.load_state_dict(torch.load(opt.load_attention))
    attention = attention.cuda()

    # backbone
    model = models.resnext101_32x8d(pretrained=True)
    model.load_state_dict(torch.load(opt.load_model))
    model = model.cuda()

    # start project
    model.eval()
    attention.eval()
    proj_trans = tv.transforms.Resize((50,50))

    if opt.vis:
        sample_folder = os.path.join(opt.output_dir, 'sample_vis')
        os.makedirs(sample_folder, exist_ok=True)

    count = 0
    FEATs = []
    PIDs = []
    Labels = []
    IMG_PATHS = []
    print("% Start feature extract !!")
    with torch.no_grad():
        for images, pids, img_pths in tqdm.tqdm(test_loader):
            images = images.cuda()
            x_result = attention(images) # [vae, input, mu, log_var, enhance, residual]
            feature = get_feature(model, x_result[4])
            FEATs.append(feature.detach().cpu())
            for img, pid, pth in zip(images, pids, img_pths):
                PIDs.append(pid)
                Labels.append(proj_trans(img).detach().cpu().unsqueeze(0))
                IMG_PATHS.append(pth)

            if count % 200 == 0:
                if opt.vis:
                    vis_sample(input = x_result[1].cpu().numpy()[0].transpose((1,2,0)), 
                                vae = x_result[0].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                residual = x_result[4].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                enhance = x_result[1].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                save_dir=os.path.join(sample_folder, img_pths[0].split('/')[-1]))
        
        FEATs = torch.cat(FEATs, dim=0)
        Labels = torch.cat(Labels, dim=0)

        # add projector
        print("% Start projectoion !!")
        tb_pth = os.path.join(opt.output_dir, 'tensorboardx/')
        if not os.path.exists(tb_pth):
            os.makedirs(tb_pth)
        print("PROJ.FEATs:",FEATs.size())
        print("PROJ.PIDs:",len(PIDs))
        print("PROJ.Label_imgs:",Labels.size())
        writer = tbx.SummaryWriter(tb_pth)
        writer.add_embedding(FEATs, metadata=PIDs, label_img=Labels)
        writer.close()  

        # ranking
        print("% Start ranking !!")
        rank_pth = os.path.join(opt.output_dir, 'ranking/')
        if not os.path.exists(rank_pth):
            os.makedirs(rank_pth)
        QUERY = FEATs
        GALLERY = FEATs
        distmat = euclidean_dist(QUERY, GALLERY)
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1) # 得到gallery sort index順序
        sample_rankidxs = np.random.randint(len(FEATs), size=20)
        
        for qid in sample_rankidxs:
            rank_list_to_im(indices[qid][1:11], IMG_PATHS[qid], IMG_PATHS, n_row=-1, n_col=-1, save_dir=os.path.join(rank_pth, 'sample_e%d.png'%qid))
        
        print("Finish!!")