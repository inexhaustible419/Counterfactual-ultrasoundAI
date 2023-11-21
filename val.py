import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as albu
from albumentations.augmentations import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from dataset_albu import Dataset as Datasetalbu
#from albumentations.core.composition import Compose
from masking_generator import *

from metrics import iou_score
from utils import AverageMeter

import pandas as pd
from skimage.transform import resize
from scipy import interpolate
from scipy import signal
from skimage.morphology import skeletonize,medial_axis


from sklearn.metrics.pairwise import euclidean_distances

import re

from torch.utils.data.dataset import ConcatDataset

"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="dsb2018_96_NestedUNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def mean_sum_of_distance(y_true, y_target):
    pairwise_distance = euclidean_distances(y_true, y_target)
    distance_sum = np.sum(pairwise_distance.min(axis=0)) + np.sum(pairwise_distance.min(axis=1))
    mean_distance = distance_sum / (len(y_true) + len(y_target))
    return mean_distance

def get_resize(prediction):
    # 截取图片的中间部分放入模型，这里是图片截取的四个边界值
    # set the boundary for cropping
    boundary = [97, 362, 95, 507]
    output_size = [boundary[1] - boundary[0], boundary[3] - boundary[2]]
    # 输出结果有100个点
    points = 100
    prediction = resize(prediction, output_size)
    return prediction

def get_skeleton(prediction):
    # 截取图片的中间部分放入模型，这里是图片截取的四个边界值
    # set the boundary for cropping
    boundary = [97, 362, 95, 507]
    output_size = [boundary[1] - boundary[0], boundary[3] - boundary[2]]
    # 输出结果有100个点
    points = 100


    #prediction = torch.Tensor.resize_(output.squeeze(0), output_size)
    #prediction = np.squeeze(output)
    # 把预测的白色像素转换成一个像素粗细的细线条
    skeleton = skeletonize(np.where(prediction > 0.6, 1, 0))
    #skeleton= medial_axis(np.where(prediction > 0.1, 1, 0))

    # 提取出曲线的坐标
    index = np.argwhere(skeleton == True)
    # sort the spline to be strictly increasing for interpolation
    # 将样条曲线排序为严格递增插值
    sorted_ind = index[index[:, 1].argsort()]
    # 插值
    f = interpolate.interp1d(sorted_ind[:, 1], sorted_ind[:, 0], kind='nearest',
                             fill_value='extrapolate')
    # get 100-point representation
    x = np.linspace(sorted_ind[:, 1][0], sorted_ind[:, 1][-1], points)
    # initialize the spline
    init = np.array([x, f(x)]).T
    # 平滑； 本来是打算直接用这个B-spline生成平滑曲线，但是有时候提取出来的曲线不是单调递增的，程序会报错，所以就前面加了一个插值
    spline = interpolate.UnivariateSpline(init[:, 0], init[:, 1])
    s = np.array([x, spline(x)]).T
    # 转换成原来图片的坐标
    # convert the coordinate back to fit the original image
    s[:, 0] = s[:, 0] + boundary[2]-1
    s[:, 1] = s[:, 1] + boundary[0]+2.5
    # 保存曲线坐标到数据框
    # ac = cp.get_active_contour(s,img,smooth=False)
    spline = pd.DataFrame(s, columns=['x', 'y'])
    spline['uniqueframe'] = pd.Series([index for i in range(100)])

    return skeleton,spline


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])


    val_transform = Compose([
        #MaskImage(),
        #DropoutImage(),
        ToPILImage(),
        Resize([config['input_h'], config['input_w']]),
        ToTensor(),
        #RandomErasing(),
        Normalize([0.0793, 0.0793, 0.0793], [0.1744, 0.1744, 0.1744]),
    ])


    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=True)

    avg_meter = AverageMeter()
    avg_msd=AverageMeter()
    avg_msd2 = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
        os.makedirs(os.path.join('outputs', config['name'], 'skeleton'), exist_ok=True)


    labels = pd.read_csv(os.path.join('splines.csv'), sep=',', header='infer')
    pattern = re.compile(r'(p\d\d)_(\d+)')

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            #output=output.cpu().numpy()
            target = torch.sigmoid(target).cpu().numpy()
            #target = target.cpu().numpy()

            # initiate an empty dataframe 初始化一个空的数据框
            cnn_prediction = pd.DataFrame(columns=['x', 'y', 'uniqueframe'])
            target_cnn_prediction = pd.DataFrame(columns=['x', 'y', 'uniqueframe'])
            labels_spline_all = pd.DataFrame(columns=['x_raw', 'y_raw'])

            cnn_prediction_resize=pd.DataFrame(columns=['x', 'y', 'uniqueframe'])

            #for i in range(len(output)):
            for i in range(1):
                for c in range(config['num_classes']):
                    #保存输出图像
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),(output[i, c] * 255).astype('uint8'))

                    prediction = np.squeeze(output[i, c])
                    #骨架算法提取识别区域骨架
                    skeleton,spline=get_skeleton(prediction)
                    # cnn_prediction = cnn_prediction.append(spline)
                    cnn_prediction = pd.concat([cnn_prediction, spline], ignore_index=True)
                    ##resize前的骨架，与target计算图像层面的msd

                    ##resize后的骨架，与标签能够计算msd
                    prediction_resize=get_resize(prediction)
                    skeleton_resize, spline_resize = get_skeleton(prediction_resize)
                    cnn_prediction_resize = pd.concat([cnn_prediction_resize, spline_resize], ignore_index=True)


                    subject = re.match(pattern, meta['img_id'][i]).group(1)
                    frame_index = re.match(pattern, meta['img_id'][i]).group(2)
                    labels_spline = labels.loc[labels['participant'] == subject]
                    labels_spline = labels_spline.loc[labels_spline['vframe'] == int(frame_index)]
                    labels_spline_all = pd.concat([labels_spline_all, labels_spline[['x_raw', 'y_raw']]], ignore_index=True)


                    target_skeleton, target_spline=get_skeleton(np.squeeze(target[i, c]))
                    # cnn_prediction = cnn_prediction.append(spline)
                    target_cnn_prediction = pd.concat([target_cnn_prediction, target_spline], ignore_index=True)

                    #保存骨架输出图像
                    cv2.imwrite(os.path.join('outputs', config['name'], 'skeleton', meta['img_id'][i] + '.jpg'),(skeleton * 255).astype('uint8'))

            msd = mean_sum_of_distance(cnn_prediction_resize[['x', 'y']], labels_spline_all)
            #print(msd)
            avg_msd.update(msd, input.size(0))

            msd_image = mean_sum_of_distance(cnn_prediction[['x', 'y']], target_cnn_prediction[['x', 'y']])
            avg_msd2.update(msd_image, input.size(0))

    print('IoU: %.4f' % avg_meter.avg)

    #cnn_prediction.to_csv(os.path.join('outputs', config['name'], 'skeleton_output.csv'))
    #target_cnn_prediction.to_csv(os.path.join('outputs', config['name'], 'target_skeleton_output.csv'))

    print('MSD: %.4f' % avg_msd.avg)
    print('MSD—Image: %.4f' % avg_msd2.avg)

    plot_examples(input, target,meta, model,num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay,meta, model,num_examples=8):

    fig, ax = plt.subplots(nrows=num_examples, ncols=5, figsize=(18,4*num_examples))
    m = datax.shape[0]
    np.random.seed(5)
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        print(meta['img_id'][image_indx])
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        imm=((np.transpose(datax[image_indx].cpu().numpy(), (1,2,0)))).astype('uint8')
        ax[row_num][0].imshow(imm)

        #imm = cv2.resize(imm, (960, 960))* 255
        #cv2.imwrite(os.path.join(str(row_num) + 'target-RandomErasing().jpg'), imm)

        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        #ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][3].imshow(np.transpose(datay[image_indx], (1, 2, 0))[:, :, 0])
        ax[row_num][3].set_title("Target image")

        #prediction = resize(np.squeeze(image_arr), output_size)
        #prediction = np.squeeze(image_arr)
        # 把预测的白色像素转换成一个像素粗细的细线条
        #skeleton = skeletonize(np.where(prediction > 0.5, 1, 0))

        skeleton,spline=get_skeleton(np.squeeze(image_arr))
        target_skeleton,_=get_skeleton(np.squeeze(datay[image_indx]))

        #im = np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0]
        #result=skeleton.astype('float32')/255+im
        #ax[row_num][0].plot(spline['x'], spline['y'], 'ro', lw=3)
        # 画图并保存
        # plot every Nth frame for inspection alpha=0.1,cmap=plt.cm.gray
        ax[row_num][2].imshow(skeleton,cmap=plt.cm.gray)
        ax[row_num][2].set_title("Segmented Image skeleton")
        ax[row_num][4].imshow(target_skeleton, cmap=plt.cm.gray)
        ax[row_num][4].set_title("Target Image skeleton")
    plt.savefig('%s.jpg' % parse_args().name)
    plt.show()



if __name__ == '__main__':
    main()