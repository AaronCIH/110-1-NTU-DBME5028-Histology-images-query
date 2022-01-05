import argparse
import torch
from dataset import query_dataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from torchvision import models
from models.vae import VanillaVAE
from functools import wraps
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os

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

def test(opt, model, attention, query_loader):
    model.eval()
    attention.eval()
    public_csv = [['query', 'prediction']]
    if opt.vis:
        sample_folder = os.path.join(opt.output_dir, 'sample_vis')
        os.makedirs(sample_folder, exist_ok=True)
    count = 1
    with torch.no_grad():
        for image1, image2, public_name in tqdm.tqdm(query_loader):
            image1, image2 = image1.cuda(), image2.cuda()
            x1_result = attention(image1) # [vae, input, mu, log_var, enhance, residual]
            x2_result = attention(image2) # [vae, input, mu, log_var, enhance, residual]            

            feature1 = get_feature(model, x1_result[4])
            feature2 = get_feature(model, x2_result[4]) 
            loss = loss_fn(feature1, feature2).cpu().detach().numpy()

            for i, dis in enumerate(loss):
                if dis > opt.threshold:
                    label = '0'
                else:
                    label = '1'
                
                public_csv.append([public_name[i], label])
            count += 1
            if count % 50 == 0:
                if opt.vis:
                    vis_sample(input = x1_result[1].cpu().numpy()[0].transpose((1,2,0)), 
                                vae = x1_result[0].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                residual = x1_result[4].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                enhance = x1_result[1].detach().cpu().numpy()[0].transpose((1,2,0)), 
                                save_dir=os.path.join(sample_folder, 'sample_ep%d'%count))
                

    np.savetxt(os.path.join(opt.output_dir, opt.csv_name), public_csv, fmt='%s', delimiter=',')
            
def validation(opt, model, attention, val_loader):
    model.eval()
    attention.eval()
    loss_list = []
    label_list = []

    max_acc = 0.
    best_threshold = 0.
    with torch.no_grad():
        for image1, image2, label in val_loader:
            image1, image2 = image1.cuda(), image2.cuda()
            x1_result = attention(image1) # [vae, input, mu, log_var, enhance, residual]
            x2_result = attention(image2) # [vae, input, mu, log_var, enhance, residual]            

            feature1 = get_feature(model, x1_result[4])
            feature2 = get_feature(model, x2_result[4])    
            loss = loss_fn(feature1, feature2).cpu().detach().numpy()

            for i, dis in enumerate(loss):
                loss_list.append(dis)
                label_list.append(label[i])

    label_list = np.array(label_list)
    num_label = len(label_list)
    loss_list = np.array(loss_list)
    loss_th_list = np.arange(0, 2, 0.01)

    # Threshold search
    for loss_th in loss_th_list:
        pred = [1 if loss <= loss_th else 0 for loss in loss_list]
        pred = np.array(pred)
        correct = np.equal(pred, label_list).sum()
        acc = round((correct / num_label) * 100, 2)

        if acc > max_acc:
            max_acc = acc
            best_threshold = loss_th
    print('best_threshold: %f \t max_acc:%.2f' % (best_threshold, max_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """train_model"""
    parser.add_argument("--load_model", type=str, default='./checkpoints/resnext101/model_epoch46_threshold_0.52_acc85.48.pth', help="path to the model of generator")
    parser.add_argument("--load_attention", type=str, default='./checkpoints/resnext101/attention_epoch46_threshold_0.52_acc85.48.pth', help="path to the model of generator")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--threshold", type=float, default=0.52, help="threshold for classification")
    parser.add_argument("--image_size", type=float, default=224, help="size of image")
    """base_options"""
    parser.add_argument("--data", type=str, default='../Dataset', help="path to dataset")
    parser.add_argument("--output_dir", type=str, default='./result/', help="name of saved model name")
    parser.add_argument("--csv_name", type=str, default='prediction.csv', help="name of saved model name")
    parser.add_argument("--gpu_id", type=str, default='0', help="gpu id")
    parser.add_argument('--lamda', default=0.001, type=float, help='Lmse + \lamda * Lkl')
    parser.add_argument('-v', '--vis', action='store_true', default=False, help='Model vae module')
    opt = parser.parse_args()
    
    query_data = query_dataset(opt.data, opt.image_size, 'test')
    query_loader = DataLoader(query_data, batch_size=128, shuffle=False, num_workers=opt.n_cpu)

    os.makedirs(opt.output_dir, exist_ok=True)

    # attention
    attention = VanillaVAE(in_channels=3,
                        latent_dim=1024,
                        hidden_dims=None,
                        W_Lkld=opt.lamda).cuda()
    attention.load_state_dict(torch.load(opt.load_attention))
    attention = attention.cuda()
    print("ATTENTION.ALPHA:",attention.recon_alpha)

    # backbone
    model = models.resnext101_32x8d(pretrained=True)
    model.load_state_dict(torch.load(opt.load_model))
    model = model.cuda()

    test(opt, model, attention, query_loader)
    # validation(opt, model, attention, query_loader)

'''
python inference.py -v --load_model ./checkpoints/resnext101/model_epoch46_threshold_0.52_acc85.48.pth --load_attention ./checkpoints/resnext101/attention_epoch46_threshold_0.52_acc85.48.pth --threshold 0.52
'''