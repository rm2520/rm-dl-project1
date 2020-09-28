from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
import multiprocessing

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=384,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=192,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=1400, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=8, type=int,

                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=2,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a1', '--arch1', type=str, default='base', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('-a2', '--arch2', type=str, default='classifier', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")

parser.add_argument('-p1', '--part1', type=int, default='6', help="6, 3, 4,12")
parser.add_argument('-p2', '--part2', type=int, default='12', help="6, 3, 4,12")

parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=80, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='/home/jiyang/Workspace/Works/video-person-reid/3dconv-person-reid/pretrained_models/resnet-50-kinetics.pth', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    #use_gpu=False
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = False


    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch1))
    if args.arch1=='resnet503d':
        model = resnet3d.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        state_dict = {}
        for key in checkpoint['state_dict']:
            if 'fc' in key: continue
            state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict, strict=False)
    else:
        base_model = models.init_model(name=args.arch1, num_classes=dataset.num_train_pids,part1=args.part1,part2=args.part2, loss={'xent','htri'})
        classifier_model = models.init_model(name=args.arch2,Feat_dim=1024,num_classes=dataset.num_train_pids,part1=args.part1,part2=args.part2)


    print("Model size: {:.5f}M".format(sum(p.numel() for p in base_model.parameters())/1000000.0))

    criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_htri = TripletLoss(margin=args.margin)

    optimizer1= torch.optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(classifier_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.stepsize, gamma=args.gamma)
        scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if use_gpu:
        base_model = nn.DataParallel(base_model).cuda()
        classifier_model = nn.DataParallel(classifier_model).cuda()

    if args.evaluate:
        print("Evaluate only")
        test(base_model,classifier_model, queryloader, galleryloader, args.pool, use_gpu)
        return

    start_time = time.time()
    print("process start time ==> Total elapsed time (h:m:s): {}".format(start_time))

    best_rank1 = -np.inf
    if args.arch1=='resnet503d':
        torch.backends.cudnn.benchmark = False
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        
        train(base_model,classifier_model, criterion_xent, criterion_htri, optimizer1,optimizer2, trainloader, use_gpu)
        
        if args.stepsize > 0:
            scheduler1.step()
            scheduler2.step()
        
        if epoch > 0 and (epoch+1) % args.eval_step == 0 or((epoch+1) ==1) or (epoch+1) == args.max_epoch:
            test_time = time.time()
            print("Test start Time ==>".format(test_time))
            print("==> Test")
            rank1 = test(base_model,classifier_model, queryloader, galleryloader, args.pool, use_gpu)
            elapsed_test = round(time.time() - test_time)
            elapsed_test = str(datetime.timedelta(seconds=elapsed_test))
            print("Test time. Total elapsed time (h:m:s): {}".format(elapsed_test))
   
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = base_model.module.state_dict()
                state_dict = classifier_model.module.state_dict()
            else:
                state_dict = base_model.state_dict()
                state_dict = classifier_model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(base_model,classifier_model, criterion_xent, criterion_htri, optimizer1,optimizer2, trainloader, use_gpu):
    base_model.train()
    classifier_model.train()

    losses = AverageMeter()

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        n, s, c, h, w = imgs.size()

        imgs, pids = Variable(imgs), Variable(pids)


        #for i in range(n):
        base,part1,part2=base_model(imgs)


        #base=torch.cat(base,0)
        #base=base_model(imgs)
        outputs,features,outputpart,featureparts = classifier_model(base,part1,part2,n,s)

        if args.htri_only:
            # only use hard triplet loss to train the network
            loss = criterion_htri(features, pids)
        else:
            # combine hard triplet loss with cross entropy loss
            xnet_lo_loss= criterion_xent(outputpart[0], pids)
            xtri_lo_loss = criterion_htri(featureparts[0], pids)
            tot_part=args.part1+args.part2
            for i in range(tot_part-1):
                xnet_lo_loss += criterion_xent(outputpart[i+1], pids)
                xtri_lo_loss += criterion_htri(featureparts[i+1], pids)

            #for i in range(7):
            xent_loss = criterion_xent(outputs, pids)
            #xent_loss_local = criterion_xent(outputslo, pids)
            htri_loss = criterion_htri(features, pids)
            #htri_loss_local = criterion_htri(featureslo, pids)


        loss = xent_loss + htri_loss+xnet_lo_loss+xtri_lo_loss



        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def test(model,classifier_model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()
    classifier_model.eval()

    qf, q_pids, q_camids = [], [], []
    tot_part = args.part1 + args.part2
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):

        if use_gpu:
            imgs = imgs[:, :40, :, :, :, :]
            imgs = imgs.cuda()
        with torch.no_grad():
            #imgs = Variable(imgs)
            # b=1, n=number of clips, s=16
            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            features=[]

            m=n
            if n>40:
                m=40



            for i in range(m):
               
                b,parts1,parts2=model(imgs[:, i,:, :, :, :])
                features.append(classifier_model(b,parts1,parts2,1,s))

            features = torch.cat(features, 0)
            features = features.data.cpu()
            features = features.view(m, 1024, tot_part+1)


            fnorm = torch.norm(features, p=2, dim=1, keepdim=True) * np.sqrt(tot_part+1)
            features = features.div(fnorm.expand_as(features))
            features = features.view(features.size(0), -1)





            features = torch.mean(features, 0)

            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs[:, :80, :, :, :, :]
            imgs = imgs.cuda()
        with torch.no_grad():
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            features = []
            for i in range(n):
                
                b, parts1,parts2 = model(imgs[:, i, :, :, :, :])
                features.append(classifier_model(b, parts1,parts2, 1, s))

            features = torch.cat(features, 0)
            features = features.view(n, 1024, tot_part+1)
            fnorm = torch.norm(features, p=2, dim=1, keepdim=True) * np.sqrt(tot_part+1)
            features = features.div(fnorm.expand_as(features))
            features = features.view(features.size(0), -1)

            
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_( qf, gf.t(),beta=1,alpha=-2)
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

if __name__ == '__main__':
    main()







