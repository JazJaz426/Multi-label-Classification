"""Train CIFAR10 with PyTorch."""
import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from randaugment import RandAugment
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import utils
from dataset import ImageCaptionDataset
from model import build_img_model, build_text_model
from utils import progress_bar, seed_everything

###################################################Arguments############################################################
parser = argparse.ArgumentParser(description='COMP5329')
parser.add_argument('--exp_name', default='default', type=str, help='experiment name')
parser.add_argument('--epochs', default=40, type=int, help='epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--img_size', default=224, type=int, help='img size')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--step_size', default=10, type=int, help='lr decay step size')
parser.add_argument('--gamma', default=0.1, type=float, help='lr decay gamma')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--seed', default=3407, type=int, help='random seed')  # arXiv:2109.08203
parser.add_argument('--model', default='resnet18', type=str, help='model name')
parser.add_argument('--weighted_loss', default=False, type=bool, help='weighted loss')

parser.add_argument('--mixup', default=False, type=bool, help='mixup augmentation')
parser.add_argument('--mixup_alpha', default=0.4, type=float, help='mixup alpha')

parser.add_argument('--autoaug', default=False, type=bool, help='auto augmentation')

parser.add_argument('--focal_loss', default=False, type=bool, help='focal loss')

parser.add_argument('--freeze_epochs', default=0, type=int, help='freeze pretrained layers for N epochs')
parser.add_argument('--after_freeze_lr', default=0.00001, type=float, help='freeze pretrained layers for N epochs')

parser.add_argument('--test', default=None, type=str, help='test model path')
parser.add_argument('--load', default=None, type=str, help='load model path')
parser.add_argument('--load_combine', default=None, type=str, help='load combined model path')

parser.add_argument('--tta', default=False, type=bool, help='test time augmentation')
parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing')

parser.add_argument('--kd_alpha', default=0, type=float, help='knowledge distillation alpha')
parser.add_argument('--kd_temp', default=2, type=float, help='knowledge distillation temperature')
parser.add_argument('--kd_teacher', default="swin_large_patch4_window7_224_linear_aug4_best.pth", type=str,
                    help='kd teacher checkpoint')

parser.add_argument('--bilinear_text', default=False, type=bool, help='bert bilinear combination')
parser.add_argument('--concat_text', default=False, type=bool, help='bert concat combination')

parser.add_argument('--threshold', default=0.5, type=float, help='threshold for classification')

args = parser.parse_args()

# set random seed to make the results reproducible
seed_everything(args.seed)

# wandb logger
wandb.init(project="asm2", entity="kd-is-the-best", name=args.exp_name)
wandb.config = {
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "seed": args.seed,
    "step_size": args.step_size,
    "gamma": args.gamma
}

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_f1 = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4641, 0.4491, 0.4220], std=[0.2371, 0.2319, 0.2357]),
])

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(args.img_size),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4641, 0.4491, 0.4220], std=[0.2371, 0.2319, 0.2357]),
]) if args.autoaug else transform_train

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4641, 0.4491, 0.4220], std=[0.2371, 0.2319, 0.2357]),
])

dataset = ImageCaptionDataset("./data/", 'train', transform=transform_train)

# random split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

# train val data loader
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=64, pin_memory=True)
val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=64, pin_memory=True)

# test data loader
testset = ImageCaptionDataset("./data/", 'test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=64)

# TTA test data loader
testset_tta = ImageCaptionDataset("./data/", 'test', transform=transform_train, tta=True)
test_tta_loader = torch.utils.data.DataLoader(testset_tta, batch_size=args.batch_size, shuffle=False, num_workers=64)

# unused code for get class weight, which can used for weighted loss
num_samples = torch.tensor([0., 18249., 939., 3499., 1014., 900., 1118., 986., 1776.,
                            842., 1182., 480., 0., 482., 201., 1543., 888., 1138.,
                            1231., 809.], dtype=torch.float)
label_weights = 1 / num_samples
label_weights[label_weights == float("Inf")] = 0
label_weights = label_weights / label_weights.sum() * 10
label_weights = label_weights.to(device)

print("class weights: ", label_weights)

# Build model
print('==> Building model..')
net, netT = build_img_model(args)
net = net.to(device)
netT = netT.to(device) if args.kd_alpha else None
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    netT = torch.nn.DataParallel(netT) if args.kd_alpha else None
    cudnn.benchmark = True



if args.load:
    # Load checkpoint for multimodal training
    print('==> Loading from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join('checkpoint', args.load))
    net.load_state_dict(checkpoint['net'])

if args.kd_alpha > 0:
    # load teacher model if kd in use
    print('==> Loading teacher from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join('checkpoint', args.kd_teacher))
    netT.load_state_dict(checkpoint['net'])

# Building text model and combined model if specified
text_model, net = build_text_model(args, device, net)
if args.load_combine:
    net.load_state_dict(torch.load(os.path.join('checkpoint', args.load_combine))['net'])

if args.test:
    # Load checkpoint for test
    print('==> Loading from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join('checkpoint', args.test))
    net.load_state_dict(checkpoint['net'])

# Loss functions for many cases
criterion = nn.BCEWithLogitsLoss(reduction='none') if args.weighted_loss else nn.BCEWithLogitsLoss()
criterion = utils.FocalLoss() if args.focal_loss else criterion
criterion_T = nn.KLDivLoss(reduction='batchmean')
criterion = criterion.to(device)

# Optimizer and Scheduler
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

wandb.watch(net)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    training_f1 = 0
    total_batch = 0

    # Freeze the pretrained model in the first n epoch if specified
    if epoch >= args.freeze_epochs:
        for param in net.parameters():
            param.requires_grad = True

    # Unfreeze the pretrained model at the n epoch and set the lr
    if epoch == args.freeze_epochs and args.freeze_epochs != 0:
        for param in optimizer.param_groups:
            param['lr'] = args.after_freeze_lr

    # Training loop
    for batch_idx, (image_file, image, caption, labels) in enumerate(tqdm(train_loader, leave=False)):
        # img tensors
        labels = smooth(labels, args.label_smoothing) if args.label_smoothing else labels
        inputs, targets = image.to(device), labels.to(device)

        # text tensors
        if args.bilinear_text or args.concat_text:
            text_input, text_masks = caption[:2]
            text_input = text_input.to(device)
            text_mask = text_masks.to(device)

        # mixup data
        if args.mixup:
            inputs, targets_a, targets_b, lam = utils.mixup_data(inputs, targets,
                                                                 args.mixup_alpha)

        # KD teacher output
        if args.kd_alpha > 0:
            with torch.no_grad():
                soft_target, pred_T = netT(inputs)
                outputs_T = F.softmax(soft_target / args.kd_temp, dim=1)

        optimizer.zero_grad()
        outputs, preds = net(inputs, text_input, text_mask) if args.bilinear_text or args.concat_text else net(inputs)

        # generate predicted label by a threshold
        preds = torch.sigmoid(outputs) > args.threshold

        # loss: BCE / KLDiv / mixup criterion
        loss = criterion(outputs, targets) if not args.mixup else utils.mixup_criterion(criterion, outputs, targets_a,
                                                                                        targets_b, lam)
        loss_T = criterion_T(F.log_softmax(outputs / args.kd_temp, dim=1), outputs_T) if args.kd_alpha > 0 else 0

        loss = (1 - args.kd_alpha) * loss + args.kd_alpha * loss_T if args.kd_alpha > 0 else loss

        if args.weighted_loss:
            loss = (loss * label_weights).mean()

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # calculate f1 score
        if args.mixup:
            training_f1 += (lam * f1_score(preds.cpu(), targets_a.cpu(), average='samples')) + (1 - lam) * f1_score(
                preds.cpu(), targets_b.cpu(), average='samples')
        else:
            training_f1 += f1_score(preds.cpu(), targets.cpu(), average='samples')
        total_batch += 1

    # log the training metrics
    wandb.log({"train_loss": train_loss / total_batch, "train_f1": training_f1 / total_batch,
               "lr": np.mean([param_group['lr'] for param_group in optimizer.param_groups])})
    print("-------------------------------------------------------")
    print("Epoch: {}/{} Training Loss: {}, Training F1: {}".format(epoch, args.epochs, train_loss / total_batch,
                                                                   training_f1 / total_batch))


def val(epoch):
    global best_f1, best_epoch
    net.eval()
    val_loss = 0
    val_f1 = 0
    total_batch = 0

    with torch.no_grad():
        # Validation loop
        for batch_idx, (image_file, image, caption, labels) in enumerate(tqdm(val_loader, leave=False)):
            # img tensors
            inputs, targets = image.to(device), labels.to(device)

            # text tensors
            if args.bilinear_text or args.concat_text:
                text_input, text_masks = caption[:2]
                text_input = text_input.to(device)
                text_mask = text_masks.to(device)

            # preds and output of the model
            outputs, preds = net(inputs, text_input, text_mask) if args.bilinear_text or args.concat_text else net(
                inputs)

            # prediction by a threshold
            preds = torch.sigmoid(outputs) > args.threshold

            # calculate loss
            loss = criterion(outputs, targets)
            if args.weighted_loss:
                loss = (loss * label_weights).mean()

            val_loss += loss.item()

            val_f1 += f1_score(preds.cpu(), targets.cpu(), average='samples')
            total_batch += 1

        # log and print the validation metrics
        f1 = val_f1 / total_batch
        print("-------------------------")
        print("Epoch: {}/{} Val Loss: {}, Val F1: {}".format(epoch, args.epochs, val_loss / total_batch, f1))
        wandb.log({"val_loss": val_loss / total_batch, "val_f1": val_f1 / total_batch})

    # Save best checkpoint.
    if f1 > best_f1:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'f1': f1,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.exp_name + '_best.pth')
        best_f1 = f1
        best_epoch = epoch
    wandb.log({"best_f1": best_f1, "best_epoch": best_epoch})


def test():
    """
    The test function do inference on the training set and write the results to the csv file for submission.
    """
    net.eval()
    path_to_save = os.path.join('submissions', '{}.csv'.format((args.test + "_tta") if args.tta else args.test))
    with open(path_to_save, 'w') as f:
        f.write('ImageID,Labels\n')
    with torch.no_grad():
        for batch_idx, (image_file, image, caption) in enumerate(test_loader):
            inputs = image.to(device)
            outputs, preds = net(inputs)
            with open(path_to_save, 'a') as f:
                for im_id, pred in zip(image_file, preds):
                    # Decode preds
                    lbl_idxs = [idx for idx, val in enumerate(pred) if val]
                    out = [str(i) for i in lbl_idxs]
                    string = f'{im_id},{" ".join(out)}\n'
                    f.write(string)

            progress_bar(batch_idx, len(test_loader))


def tta_test():
    """
    TTA test function. Test time augmentations. Write the results to the csv file for submission.
    """
    net.eval()
    path_to_save = os.path.join('submissions', '{}.csv'.format((args.test + "_tta") if args.tta else args.test))
    with open(path_to_save, 'w') as f:
        f.write('ImageID,Labels\n')
    with torch.no_grad():
        for batch_idx, (image_file, image, caption) in enumerate(test_tta_loader):
            out_temp = []
            if args.bilinear_text or args.concat_text:
                text_input, text_masks = caption[:2]
                text_input = text_input.to(device)
                text_mask = text_masks.to(device)

            for img in image:
                inputs = img.to(device)
                # outputs, _ = net(inputs)

                outputs, preds = net(inputs, text_input, text_mask) if args.bilinear_text or args.concat_text else net(
                    inputs)
                out_temp.append(outputs.cpu().numpy())

            out_mean = np.mean(out_temp, axis=0)

            preds = torch.nn.functional.sigmoid(torch.from_numpy(out_mean)) > args.threshold

            with open(path_to_save, 'a') as f:
                for im_id, pred in zip(image_file, preds):
                    # Decode preds
                    lbl_idxs = [idx for idx, val in enumerate(pred) if val]
                    out = [str(i) for i in lbl_idxs]
                    string = f'{im_id},{" ".join(out)}\n'
                    f.write(string)

            progress_bar(batch_idx, len(test_tta_loader))


def smooth(y, eps=0.4):
    """
    label smoothing.
    """
    a = 1 - eps * (1 + 1 / y.shape[1])
    b = eps / y.shape[1]
    return a * y + b


# ----------------------------------------------------------------------------------------------------------------------
# epoch loop, main function
for epoch in trange(start_epoch, args.epochs, leave=False):
    if not args.test:
        train(epoch)
        val(epoch)
        scheduler.step()
    else:
        tta_test()
        break
