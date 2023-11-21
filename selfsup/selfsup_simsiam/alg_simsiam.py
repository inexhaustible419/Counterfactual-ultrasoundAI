from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn import functional as F

from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

from models import SimpleCNN

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ATBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sig=nn.Sigmoid()
        self.Maxpool=nn.AdaptiveMaxPool3d((1, None, None))
        self.Avgpool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv3=nn.Conv2d(2,middle_channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        cbam=out

        avg_pool = self.Avgpool(cbam)
        max_pool = self.Maxpool(cbam)
        concat = torch.cat((avg_pool, max_pool), dim=1)

        cbam = self.conv3(concat)
        cbam = self.sig(cbam)
        out=torch.multiply(cbam, out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class SimSiam(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        #arch='resnet50',
        #hidden_mlp: int = 2048,
        #hidden_mlp: int = 6144,
        hidden_mlp: int = 4608,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = 'adam',
        exclude_bn_bias: bool = False,
        start_lr: float = 0.,
        learning_rate: float = 1e-3,
        final_lr: float = 0.,
        weight_decay: float = 1e-6,
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        #self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.init_model()

        #self.final = nn.Conv2d(nb_filter[0], kernel_size=1)

        # compute iters per epoch
        nb_gpus = len(self.gpus) if isinstance(gpus, (list, tuple)) else self.gpus
        assert isinstance(nb_gpus, int)
        global_batch_size = self.num_nodes * nb_gpus * self.batch_size if nb_gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        #if isinstance(self.arch, torch.nn.Module):
        #if isinstance(self,torch.nn.Module):
            #backbone = self.arch
            #nb_filter = [32, 64, 128, 256, 512]
        '''
        encoder = nn.Sequential(
                VGGBlock(1, nb_filter[0], nb_filter[0]),
                nn.MaxPool2d(2, 2),
                VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1]),
                nn.MaxPool2d(2, 2),
                VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2]),
                nn.MaxPool2d(2, 2),
                VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3]),
                nn.MaxPool2d(2, 2),
                VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4]),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
            )
        '''

        self.online_network = SiameseArm(
             input_dim=self.hidden_mlp, hidden_size=4096, output_dim=self.feat_dim
        )


    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        b = b.detach()  # stop gradient of backbone + projection mlp
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = -1 * (a * b).sum(-1).mean()
        return sim

    def training_step(self, batch, batch_idx):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        self.log_dict({"train_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2, _), y = batch

        # Image 1 to image 2 loss
        _, z1, h1 = self.online_network(img_1)
        _, z2, h2 = self.online_network(img_2)
        loss = self.cosine_similarity(h1, z2) / 2 + self.cosine_similarity(h2, z1) / 2

        # log results
        self.log_dict({"val_loss": loss})

        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {
                'params': params,
                'weight_decay': weight_decay
            },
            {
                'params': excluded_params,
                'weight_decay': 0.
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'lars':
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

    seed_everything(1234)

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = SimSiam.add_model_specific_args(parser)
    args = parser.parse_args()

    # pick data
    dm = None

    # init datamodule
    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=val_split,
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]
        args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = "lars"
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    model = SimSiam(**args.__dict__)

    # finetune in real-time
    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = pl.Trainer.from_argparse_args(
        args,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


from typing import Optional, Tuple

import torch
from torch import nn

from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class MLP(nn.Module):

    def __init__(self, input_dim: int =6144 , hidden_size: int = 4096, output_dim: int = 256) -> None:
        super().__init__()#6144,4096
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(
        self,
        #encoder: Optional[nn.Module] = None,
        input_dim: int = 2048,
        hidden_size: int = 4096,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        #if encoder is None:
            #encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        #self.encoder = encoder
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        '''
        self.conv0_0 = VGGBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        '''


        self.conv0_0 = ATBlock(3, nb_filter[0], nb_filter[0])
        self.conv1_0 = ATBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ATBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ATBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ATBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.flat=nn.Flatten()

        # Projector
        self.projector = MLP(input_dim, hidden_size, output_dim)
        # Predictor
        self.predictor = MLP(output_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # y = self.encoder(x)[0]
        #y = self.encoder(x)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        xx=self.pool(x4_0)
        y=self.flat(xx)
        #print(y.shape)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h


if __name__ == "__main__":
    cli_main()
