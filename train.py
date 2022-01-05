import torch
from models.byol_pytorch import BYOL
from models.vae import VanillaVAE
from torch.nn.modules.batchnorm import BatchNorm1d
from torchvision import models
from dataset import Histology_data, query_dataset
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

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

def SSL(args):
   # In[0]: Dataloader
    train_data = Histology_data(args.data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    val_data = query_dataset(args.data, args.image_size, 'val')
    val_loader = DataLoader(val_data, batch_size=32, num_workers=args.n_cpu, shuffle=False)
 
   # In[1]: model setting
    ############################################################
    ## * attention: 
    ##      - input: image
    ##      - output: [vae, input, mu, log_var, enhance, residual]
    ##      - loss: loss_function(*result, M_N=kld_weight), return: {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
    ############################################################
    attention = VanillaVAE(in_channels=3,
                            latent_dim=1024,
                            hidden_dims=None,
                            W_Lkld=args.lamda).cuda()
    print(attention.recon_alpha)
    # feature extractor backbone
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True).cuda()
    elif args.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True).cuda()
    elif args.model == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True).cuda()
    print('Using the model of %s' % (args.model))
    max_acc, best_threshold = 0, 0
    if args.load:
        print('Load pretrained model!')
        model.load_state_dict(torch.load(args.load))

    # BYOL module
    learner = BYOL(
        model,
        image_size = args.image_size,
        hidden_layer = 'avgpool',
        projection_size = 512,           # the projection size
        projection_hidden_size = 2048,   # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay = 0.996      # the moving average decay factor for the target encoder, already set at what paper recommends
    )

    # optimizer
    if args.opt == 'adam':
        opt = torch.optim.Adam([{'params':attention.parameters()}, {'params':learner.parameters()}], lr=args.lr, weight_decay=4e-4)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD([{'params':attention.parameters()}, {'params':learner.parameters()}], lr=args.lr, momentum=0.9, weight_decay=1.5e-6)

   # Start training
    print('Start training!')
    W_Lconst = args.eta
    if args.vis:
        sample_folder = os.path.join(args.save_model_path, 'sample_vis')
        os.makedirs(sample_folder, exist_ok=True)

    for epoch in range(args.n_epochs):
        total_loss = 0.
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for x1, x2 in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()
            # STEP1. VAE
            x1_result = attention(x1) # [vae, input, mu, log_var, enhance, residual]
            x2_result = attention(x2) # [vae, input, mu, log_var, enhance, residual]
            Lvae_dict1 = attention.loss_function(*x1_result) # {'loss', 'Reconstruction_Loss', 'KLD'}
            Lvae_dict2 = attention.loss_function(*x2_result) # {'loss', 'Reconstruction_Loss', 'KLD'}
            # STEP2. BYOL
            Lbyol = learner(x1_result[4], x2_result[4])
            # STEP3. backward + optim
            opt.zero_grad()
            L_total = Lbyol + W_Lconst*(Lvae_dict1['loss']+Lvae_dict2['loss'])/2
            L_total.backward()
            total_loss += L_total.item()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            pbar.update()
            pbar.set_postfix(
                Lall=f"{total_loss:.4f}",
                Ltotal=f"{L_total.item():.4f}",
                Lbyol=f"{Lbyol.item():.4f}",
                Lconst=f"{Lvae_dict1['Reconstruction_Loss'].item():.4f}",
                Lkdl=f"{Lvae_dict1['KLD'].item():.4f}",
            )
        pbar.close()
        # STEP4. vis sample
        print(attention.recon_alpha)
        if args.vis:
            vis_sample(input = x1_result[1].cpu().numpy()[0].transpose((1,2,0)), 
                    vae = x1_result[0].detach().cpu().numpy()[0].transpose((1,2,0)), 
                    residual = x1_result[4].detach().cpu().numpy()[0].transpose((1,2,0)), 
                    enhance = x1_result[1].detach().cpu().numpy()[0].transpose((1,2,0)), 
                    save_dir=os.path.join(sample_folder, 'sample_ep%d'%epoch))
        # STEP5. Validation
        threshold, val_acc = validation(args, model, attention, val_loader, epoch)
        if val_acc > max_acc:
            max_acc = val_acc
            best_threshold = threshold                                                                                                  
            # save your improved networkW
            torch.save(model.state_dict(), '%s/model_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))
            torch.save(attention.state_dict(), '%s/attention_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))
            # torch.save(learner.state_dict(), '%s/BYOL_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))

        elif epoch % 5 == 0:
            torch.save(model.state_dict(), '%s/model_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))
            torch.save(attention.state_dict(), '%s/attention_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))
            # torch.save(learner.state_dict(), '%s/BYOL_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))

    torch.save(model.state_dict(), '%s/model_final_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, threshold, val_acc))
    torch.save(attention.state_dict(), '%s/attention_final_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, threshold, val_acc))
    print('Best threshold: %f \t max_acc:%.2f%%' % (best_threshold, max_acc))

def validation(args, model, attention, val_loader, epoch):
    model.eval()
    attention.eval()
    loss_list = []
    label_list = []

    max_acc = 0.
    best_threshold = 0.
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, args.n_epochs), unit=" step")
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
            pbar.update()

        label_list = np.array(label_list)
        num_label = len(label_list)
        loss_list = np.array(loss_list)
        loss_th_list = np.arange(0 ,2 , 0.01)

        # Threshold search
        for loss_th in loss_th_list:
            pred = [1 if loss <= loss_th else 0 for loss in loss_list]
            pred = np.array(pred)
            correct = np.equal(pred, label_list).sum()
            acc = round((correct / num_label) * 100, 2)

            if acc > max_acc:
                max_acc = acc
                best_threshold = loss_th
        pbar.set_postfix(
        val_acc = f"{max_acc:.2f}%",
        threshold = f"{best_threshold}"
        )
        pbar.close()
        return best_threshold, max_acc

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_epochs', default=60, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch_size for training')
    parser.add_argument('--image_size', default=224, type=int, help='size of image')
    parser.add_argument('--n_cpu', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate of the adam')
    parser.add_argument('--load', type=str, default='', help="Model checkpoint path")
    parser.add_argument('--data', default='../Dataset/', type=str, help="Training images directory")
    parser.add_argument('--model', default='resnext101', type=str, help="feature backbone(resnet18/resnext50/resnext101)")
    parser.add_argument('--opt', default='sgd', type=str, help="adam/sgd")
    parser.add_argument('--save_model_path', default='./checkpoints/resnext101', type=str, help="Path to save model")
    parser.add_argument('--lamda', default=0.001, type=float, help='Lmse + \lamda * Lkl')
    parser.add_argument('--eta', default=10, type=float, help='Ltask + \eta * Lreconstruction')
    parser.add_argument('-v', '--vis', action='store_true', default=False, help='Model vae module')
    args = parser.parse_args()
    os.makedirs(args.save_model_path, exist_ok=True)

    SSL(args)

    
'''
python train.py -v --eta 10 --save_model_path './checkpoints/res101_e10_f/'  
'''
    