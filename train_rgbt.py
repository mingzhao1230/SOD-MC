import argparse
from rgbt_dataset import Dataset
from torchvision import transforms
import transform_rgbd
from torch.utils import data
import torch
from collections import OrderedDict
from models.SalMamba_dual import Model
from datetime import datetime
import os
from thop import profile
import numpy as np
import IOU
import datetime
import torch.distributed as dist
import random
from Smeasure import S_object,S_region
from utils.init_func import group_weight
import torch.nn as nn
import logging
import ast
import torch.fft
import torch.nn.functional as F
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
p = OrderedDict()
p['lr'] = 1e-4  # Learning rate
p['wd'] = 0.01  # Weight decay
p['momentum'] = 0.90  # Momentum
showEvery = 300
CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)
# =======================================================
def weighted_mse_loss(student_feat, teacher_feat, label):
    mask = F.interpolate(label, size=student_feat.shape[2:], mode='nearest')
    weight = mask * 1.0 + (1 - mask) * 0.1
    return (weight * (student_feat - teacher_feat) ** 2).mean()

def frequency_structure_loss(student_feat, teacher_feat, weight_amp=1.0, weight_phase=1.0):
    device = student_feat.device
    s_cpu = student_feat.float().cpu()
    t_cpu = teacher_feat.float().cpu()
    
    fft_student = torch.fft.rfft2(s_cpu, norm='ortho')
    fft_teacher = torch.fft.rfft2(t_cpu, norm='ortho')
    
    amp_student = torch.log1p(fft_student.abs() + 1e-8)
    amp_teacher = torch.log1p(fft_teacher.abs() + 1e-8)
    loss_amp = torch.nn.functional.l1_loss(amp_student, amp_teacher)
    
    phase_student = fft_student.angle()
    phase_teacher = fft_teacher.angle()
    loss_phase = 1.0 - torch.cos(phase_student - phase_teacher)
    loss_phase = loss_phase.mean()
    
    total_loss_cpu = weight_amp * loss_amp + weight_phase * loss_phase
    return total_loss_cpu.to(device)
# =======================================================

def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce+iou

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
print(f"CUDA Available: {torch.cuda.is_available()}")

parser.add_argument('--cuda', type=bool, default=True)

# train
parser.add_argument('--epoch', type=int, default=50) 
parser.add_argument('--epoch_save', type=int, default=10)
parser.add_argument('--save_fold', type=str, default='./checkpoints')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_thread', type=int, default=8)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--resume', type=str, default=None, help='path to resume checkpoint')
parser.add_argument('--start_epoch', type=int, default=None, help='Manual epoch number')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--test_dataset', type=list, default=['VT821'])
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

change_epoch = 30

config.save_fold = config.save_fold + '/' + 'Samba/rgbt_datasets'
if not os.path.exists("%s" % (config.save_fold)):
    os.makedirs("%s" % (config.save_fold))

def get_logger(filename, verbosity=1, name=None, mode='w'):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

log_mode = 'a' if config.resume else 'w'
logger = get_logger(os.path.join(config.save_fold, 'training_log.txt'), mode=log_mode)

best_sm = 0
best_epoch = 0

def test(test_loader, model, epoch, save_path, optimizer, scheduler, is_stage2=False):
    global best_sm, best_epoch
    model.eval()
    s_sum = 0
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            image,label, depth, name, split, size = data_batch['image'],data_batch['label'], data_batch['depth'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']

            image, depth, label = image.cuda(),depth.cuda(), label.cuda()
            if is_stage2:
                image = torch.zeros_like(image)
            out, saliency, _, _ = model(image,depth)
            
            pre1 = torch.nn.Sigmoid()(out[0])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            gt = label[0]
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            alpha = 0.5
            y = gt.mean()
            if y == 0:
                x = pre1.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pre1.mean()
                Q = x
            else:
                Q = alpha * S_object(pre1, gt) + (1 - alpha) * S_region(pre1, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_sum += Q.item()
        sm = s_sum / len(test_loader.dataset)

        stage_str = "Stage II (Single-T)" if is_stage2 else "Stage I"
        print(f'[{stage_str}] Epoch: {epoch+1} sm: {sm:.4f} (Best: {best_sm:.4f} at Epoch {best_epoch})')

        if sm >= best_sm:
            best_sm = sm
            best_epoch = epoch + 1
            if is_stage2:
                save_name = 'epoch_best_Stage2_SingleT.pth'
            else:
                save_name = 'epoch_best_Stage1.pth'
        
            torch.save(model.state_dict(), '%s/%s' % (save_path, save_name))
            print(f'>>> Save Best Model (Weights Only): {save_name} <<<')

        return sm


if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        transform_rgbd.RandomFlip(),
        transform_rgbd.RandomRotate(),
        transform_rgbd.colorEnhance(),
        transform_rgbd.randomPeper(),
        transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbd.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform_rgbd.ToTensor()])
    dataset_train = Dataset(datasets=['VT_train'],transform=composed_transforms_ts, mode='train')

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread,
                                 drop_last=True,
                                 shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))

    composed_transforms_te = transforms.Compose([
    transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
    transform_rgbd.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform_rgbd.ToTensor()])

    dataset_test = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
    test_loader = data.DataLoader(dataset_test, batch_size=1, num_workers=config.num_thread,drop_last=True, shuffle=False)
    print("Testing Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_test), len(test_loader)))

    model = Model()
    if config.cuda:
        model = model.cuda()
    
    print(">>> Initializing: Freezing Control Branch for Stage I default <<<")
    for param in model.backbone_ctrl.parameters():
        param.requires_grad = False
    for param in model.zero_conv.parameters():
        param.requires_grad = False


    if hasattr(model.backbone, 'clip_guide') and model.backbone.clip_guide is not None:
        print(">>> Configuring Main CLIP for Stage I (Unfreeze Adapter) <<<")
        clip_main = model.backbone.clip_guide
        
        for param in clip_main.clip_model.parameters():
            param.requires_grad = False
        
        for param in clip_main.context_adapter.parameters():
            param.requires_grad = True
        
        if hasattr(clip_main, 'prompt_weights'):
            clip_main.prompt_weights.requires_grad = True

    if hasattr(model.backbone_ctrl, 'clip_guide') and model.backbone_ctrl.clip_guide is not None:
        print(">>> Configuring Control CLIP for Stage I (Fully Frozen) <<<")
        clip_ctrl = model.backbone_ctrl.clip_guide
        for param in clip_ctrl.parameters(): 
            param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=p['lr'], betas=(0.9, 0.999),weight_decay=p['wd'])

    start_epoch = 0
    loss_wirte = []
    sm_wirte = []
    

    if config.resume and os.path.isfile(config.resume):
        print(f"==> loading checkpoint '{config.resume}'")
        checkpoint = torch.load(config.resume)
        

        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            file_epoch = checkpoint['epoch']
            best_sm = checkpoint.get('best_sm', 0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"==> Checkpoint info found: Saved at Epoch {file_epoch}")
        else:
            file_epoch = 0
            best_sm = 0
            best_epoch = 0
            print("==> Checkpoint is strictly weights-only (No epoch/optimizer info).")


        if config.start_epoch is not None:
            start_epoch = config.start_epoch
            print(f"==> [Manual Override] Start epoch set to {start_epoch}")
        else:
            start_epoch = file_epoch
            if start_epoch == 0:
                print("==> Warning: Start epoch is 0. If resuming, please use --start_epoch [N]")
            else:
                print(f"==> Resuming from Epoch {start_epoch}")


        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("==> Model weights loaded successfully.")
        

        if start_epoch > change_epoch:
            print(f"Resuming in Stage II (Epoch {start_epoch}). Adjusting freeze/unfreeze status...")
            for param in model.backbone.parameters(): param.requires_grad = False
            for param in model.decoder.parameters(): param.requires_grad = False

            if hasattr(model.backbone, 'clip_guide') and model.backbone.clip_guide is not None:
                 for param in model.backbone.clip_guide.parameters():
                     param.requires_grad = False

            for param in model.backbone_ctrl.parameters(): param.requires_grad = True
            for param in model.zero_conv.parameters(): param.requires_grad = True


            if hasattr(model.backbone_ctrl, 'clip_guide') and model.backbone_ctrl.clip_guide is not None:
                for param in model.backbone_ctrl.clip_guide.parameters():
                    param.requires_grad = False
                print(">>> Resume Stage II: Control CLIP Frozen! <<<")

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=p['lr'], betas=(0.9, 0.999), weight_decay=p['wd'])
        

        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> Optimizer state loaded.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state (Ignore if fine-tuning). Error: {e}")
        
        if os.path.exists("./train_loss_rgbt.txt"):
            with open("./train_loss_rgbt.txt", 'r') as f:
                content = f.read()
                if content:
                    loss_wirte = ast.literal_eval(content)
                    print(f"==> Loaded {len(loss_wirte)} epoch loss records.")
        
        if os.path.exists("./Smeasure_rgbt.txt"):
            with open("./Smeasure_rgbt.txt", 'r') as f:
                content = f.read()
                if content:
                    sm_wirte = ast.literal_eval(content)
                    print(f"==> Loaded {len(sm_wirte)} epoch S-measure records.")
        
    if start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10, 
            eta_min = 1e-5,
            last_epoch = start_epoch - 1
        )
    if config.resume and isinstance(checkpoint, dict) and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            pass

    optimizer.zero_grad()
    accumulation_steps = 4
    iter_num = len(dataloader)
    
    for epoch in range(start_epoch, config.epoch):

        if epoch == change_epoch:
            print(f"\n{'='*20} Switching to Stage II (Epoch {epoch+1}) {'='*20}")

            torch.save(model.state_dict(), f'{config.save_fold}/epoch_{epoch}_Stage1_End.pth')
            
            best_model_path = os.path.join(config.save_fold, 'epoch_best_Stage1.pth')
            if os.path.exists(best_model_path):
                print(f">>> Loading Best Stage I Model from: {best_model_path}")
                best_ckpt = torch.load(best_model_path)
                if isinstance(best_ckpt, dict) and 'state_dict' in best_ckpt:
                    model.load_state_dict(best_ckpt['state_dict'])
                else:
                    model.load_state_dict(best_ckpt)
                print(">>> Loaded Best Weights")
            else:
                print(">>> Warning: Best Stage I model not found. Continuing with current weights.")

            best_sm = 0
            best_epoch = 0
            
            model.backbone_ctrl.load_state_dict(model.backbone.state_dict())
            print(">>> Weights copied from Backbone to Control Branch. <<<")
            
            for param in model.backbone.parameters(): param.requires_grad = False
            for param in model.decoder.parameters(): param.requires_grad = False
            

            if hasattr(model.backbone, 'clip_guide') and model.backbone.clip_guide is not None:
                for param in model.backbone.clip_guide.parameters():
                    param.requires_grad = False

            for param in model.backbone_ctrl.parameters(): param.requires_grad = True
            for param in model.zero_conv.parameters(): param.requires_grad = True


            if hasattr(model.backbone_ctrl, 'clip_guide') and model.backbone_ctrl.clip_guide is not None:
                clip_ctrl = model.backbone_ctrl.clip_guide
                for param in clip_ctrl.parameters(): 
                    param.requires_grad = False
                print(">>> Stage II Switch: Control CLIP Frozen! <<<")
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=p['lr'], betas=(0.9, 0.999), weight_decay=p['wd'])
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=10, 
                eta_min=1e-5, 
                last_epoch=-1
            )
            print(f">>> Optimization Reset. Training only Control Branch now. <<<")
            print(f"{'='*60}\n")
        # ==========================================================

        loss_all = 0
        loss_kd_sum = 0 
        
        model.zero_grad()
        optimizer.zero_grad()
        model.train()

        for i, data_batch in enumerate(dataloader):
            image, label, depth = data_batch['image'], data_batch['label'], data_batch['depth']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue
            if config.cuda:
                image, label, depth = image.cuda(), label.cuda(), depth.cuda()
            

            out, saliency, feat_teacher, feat_student = model(image, depth, epoch=epoch + 1, change_epoch=change_epoch)
            

            if epoch < change_epoch:
                
                loss = structure_loss(out, label) + structure_loss(saliency, label)
            else:
                
                loss_base = structure_loss(out, label)
                
                loss_kd = 0
                if feat_teacher is not None:
                    
                    for ft, fs in zip(feat_teacher, feat_student):
                        
                        loss_spatial = weighted_mse_loss(fs, ft, label)
                        
                        loss_freq = frequency_structure_loss(fs, ft)
                        loss_kd += (loss_spatial + 0.1 * loss_freq)
                    loss_kd_sum += loss_kd.item()
                    
                loss = loss_base + 0.1 * loss_kd 
            
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_all += loss.data
            if i % showEvery == 0:
                
                kd_info = ""
                if epoch >= change_epoch and feat_teacher is not None:
                     avg_kd = loss_kd_sum / (i + 1)
                     kd_info = f" || KD: {avg_kd:.4f}"

                log_msg = '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sum : %10.4f%s' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num, loss_all / (i + 1), kd_info)
                logger.info(log_msg) 
        
        epoch_loss = loss_all / len(dataloader)
        loss_wirte.append(epoch_loss)
        scheduler.step()

        if epoch >= 0:
            is_stage2 = (epoch >= change_epoch)
            sm = test(test_loader, model, epoch, config.save_fold, optimizer, scheduler, is_stage2=is_stage2)
            sm_wirte.append(sm)

            if (epoch + 1) % config.epoch_save == 0:

                torch.save(model.state_dict(), f'{config.save_fold}/epoch_{epoch+1}_checkpoint.pth')
        
        with open("./train_loss_rgbt.txt", 'w') as train_los:
            train_los.write(str(loss_wirte))
        with open("./Smeasure_rgbt.txt", 'w') as train_sm:
            train_sm.write(str(sm_wirte))

    torch.save(model.state_dict(), f'{config.save_fold}/Final_Model.pth')
    print(">>> Training Finished. <<<")