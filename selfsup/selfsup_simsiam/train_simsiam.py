


import os
import argparse
import logging
import time
import torch, torchvision
import torch.nn as nn
import pytorch_lightning as pl
from data import dataloader_selfsup
from models import SimpleCNN

# from selfsup.selfsup_bolts.selfsupAlgs import alg_simclr
from selfsup.selfsup_simsiam import alg_simsiam

from pl_bolts.models.self_supervised import SimCLR, BYOL,SwAV
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10, CPCTrainTransformsCIFAR10
# from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
from data.transforms_selfsup import SimCLRTrainDataTransform


def get_logger(filename='logger',level='info', outPath='./'):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    fmt = "%(asctime)s - %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S"
    format_str = logging.Formatter(fmt,datefmt)

    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    timeStr = time.strftime("%Y-%m-%d--%H-%M-%S",time.localtime())
    th = logging.FileHandler(filename=os.path.join(outPath,filename+'--'+timeStr), encoding='utf-8')
    th.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler
    logger.addHandler(th)
    return logger

def get_args():
    parser = argparse.ArgumentParser(description='self-supervised Training')

    parser.add_argument('--algorithm', type=str, default='SimSiam', help='self supervised algorithm')

    parser.add_argument('--model', default="MyModel", type=str,
                        help='model type (default: MyModel)')
    parser.add_argument('--input_channels', default=1, type=int,
                       help='channels of input images, 1: gray image; 3: pseudo colorful image')
    parser.add_argument('--imagenetpretrained', default=False, type=bool,
                        help='whether to load the imagenet pretrained params')

    parser.add_argument('--image_size', type=tuple, default=(63,412))
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=64)
    parser.add_argument('--warmup_epochs', default=3, type=int, help='total epochs to run')
    ##熊parser.add_argument('--max_epochs', default=50, type=int, help='total epochs to run')
    parser.add_argument('--max_epochs', default=26, type=int, help='total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # 6e-4
    parser.add_argument('--weight_decay', default=0.5e-3, type=float, help='weight decay')  # 0.5e-3
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)

    parser.add_argument('--seed', default=2020, type=int, help='random seed')


    args = parser.parse_args()

    logger.info(dict(args._get_kwargs()))
    return args



if __name__ == '__main__':
    savepathBase = './output_selfsup_bolts'
    if not os.path.isdir(savepathBase):
        os.makedirs(savepathBase)
    logger = get_logger(outPath=savepathBase)

    args = get_args()

    ################################################################################
    udeftransforms = SimCLRTrainDataTransform(args.image_size)

    # datafolder = '/media/xys/work/UltraSuite/selfsupImgs_3U'
    ##熊datafolder = '/media/xys/work/UltraSuite/selfsupImgs'
    datafolder = 'D:/0jia/UltraSuite/selfsupImgs_3U'
    datafolder = 'D:/0jia/UltraSuite/selfsupImgs'
    train_loader = dataloader_selfsup.build_dataloader(datafolder, args.batch_size, udeftransforms=udeftransforms)
    ############################################################################


    if args.model == 'MyModel':
        model_whole = SimpleCNN.MyModel()
        base_encoder = model_whole.backbone
    else:
        base_encoder = args.model

    selfsupmodel = alg_simsiam.SimSiam(gpus=args.num_gpus,
                                       num_samples=len(train_loader.dataset),
                                       batch_size=args.batch_size,
                                       #arch=base_encoder,
                                       #hidden_mlp=32*15*103,
                                       #hidden_mlp=6144,
                                       hidden_mlp=4608,
                                       warmup_epochs = args.warmup_epochs,
                                       max_epochs = args.max_epochs,
                                       dataset='',)

    trainer = pl.Trainer(
            gpus = args.num_gpus,
            max_epochs = args.max_epochs,
            accumulate_grad_batches = args.accumulate_grad_batches,
            sync_batchnorm = True,
            # logger=[tblogger, csvlogger],
            # amp_backend = 'apex',
            # amp_level = '01'
        )
    trainer.fit(selfsupmodel, train_loader)
    #trainer.fit(selfsupmodel, train_loader, ckpt_path='./lightning_logs/version_23/checkpoints/epoch=5-step=312.ckpt')

    modelfilename = f'./selfsup-{args.algorithm}-{args.model}.pt'
    torch.save(model_whole.state_dict(), modelfilename)

