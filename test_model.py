import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from dataset import get_test_dataset, inverse_normalize
from dataset_avs import get_ms3_dataset, get_s4_dataset
import cv2
from tqdm import tqdm
import shutil
from datetime import datetime
import model_slot
import model_baseline
import json
import random

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='noname', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--model', default=None, choices=['jsa', 'av_mil'])

    parser.add_argument('--testset', default=None, type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--test_data_path', default=None, type=str, help='Root directory path of test data')
    parser.add_argument('--test_manifest_path', default=None, type=str,
                        help='Optional CSV file listing test sample IDs')
    parser.add_argument('--test_gt_path', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Checkpoint filename inside the experiment directory')
    
    # hyper-params
    parser.add_argument('--aud_length', default=None, type=float)

    # training/evaluation parameters
    parser.add_argument('--batch_size', default=None, type=int, help='Batch Size')
    parser.add_argument("--infer_sharpening", type=float, default=0.1)
    parser.add_argument("--num_slots", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument('--wandb', type=str, default=None)

    # Distributed params
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)

    # Evaluation
    parser.add_argument('--alpha', default=0.6, type=float, help='alpha')

    args = parser.parse_args()

    with open(os.path.join(args.model_dir, args.experiment_name, 'configs.json'), 'r') as f:
        config_dict = json.load(f)
    config_namespace = argparse.Namespace(**config_dict)

    for key, value in vars(args).items():
        if value is not None:
            setattr(config_namespace, key, value)
    
    return config_namespace

def setup_seed(seed):
    print("Random seed: %d" %(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def main(args):
    setup_seed(12345)
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    from torchvision.models import resnet18
    object_saliency_model = resnet18(weights='ResNet18_Weights.IMAGENET1K_V1') # resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )

    if getattr(args, 'model', 'jsa') == 'jsa':
        audio_visual_model = model_slot.mymodel(args)
    else:
        audio_visual_model = model_baseline.AudioVisualMIL(args)
    audio_visual_model.cuda(args.gpu)
    object_saliency_model.cuda(args.gpu)

    # Load weights
    checkpoint = getattr(args, 'checkpoint', None)
    if checkpoint is None:
        checkpoint = 'final.pth' if os.path.exists(os.path.join(model_dir, 'final.pth')) \
            else '%s_best.pth' % args.testset
    ckp_fn = os.path.join(model_dir, checkpoint)
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu', weights_only=False)
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {ckp_fn}')
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    if args.testset == 'ms3':
        testdataset = get_ms3_dataset(args)
    elif args.testset == 's4':
        testdataset = get_s4_dataset(args)
    else:
        testdataset = get_test_dataset(args, args.testset)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    print("Loaded dataloader.")

    mAP_aud, auc_aud, \
    mAP_img_query, auc_img_query, \
    mAP_iqr, auc_iqr, \
    mAP_obj_prior, auc_obj_prior, \
    mAP_ogl, auc_ogl, \
    mAP_extra_iqr_ogl, auc_extra_iqr_ogl = \
    validate_img_aud(testdataloader, audio_visual_model, object_saliency_model, viz_dir, args.testset, -1, args)
    
    print('AUD_%s/cIoU, auc' %(args.testset), f'{mAP_aud:.4f}', f'{auc_aud:.4f}')
    print('IMG_QUERY_%s/cIoU, auc' %(args.testset), f'{mAP_img_query:.4f}', f'{auc_img_query:.4f}')
    print('IQR_%s/cIoU, auc' %(args.testset), f'{mAP_iqr:.4f}', f'{auc_iqr:.4f}')
    print('OBJ_PRIOR_%s/cIoU, auc' %(args.testset), f'{mAP_obj_prior:.4f}', f'{auc_obj_prior:.4f}')
    print('OGL_%s/cIoU, auc' %(args.testset), f'{mAP_ogl:.4f}', f'{auc_ogl:.4f}')
    print('EXTRA_IQR_OGL_%s/cIoU, auc' %(args.testset), f'{mAP_extra_iqr_ogl:.4f}', f'{auc_extra_iqr_ogl:.4f}')
    return

@torch.no_grad()
def validate_img_aud(testdataloader, audio_visual_model, object_saliency_model, viz_dir, testset, epoch, args):
    audio_visual_model.eval()
    object_saliency_model.eval()

    evaluator_aud = utils.Evaluator()
    evaluator_img_query = utils.Evaluator()
    evaluator_iqr = utils.Evaluator()
    evaluator_obj_prior = utils.Evaluator()
    evaluator_ogl = utils.Evaluator()
    evaluator_extra_iqr_ogl = utils.Evaluator()

    for step, (image, spec, bboxes, name, label) in enumerate(tqdm(testdataloader)):
        # Handle 5D image tensor by combining first two dimensions
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
            spec = spec.unsqueeze(0)
            bboxes = bboxes.unsqueeze(0)

        if len(image.size()) == 5:
            b, n, c, h, w = image.size()
            image = image.reshape(b*n, c, h, w)
            b, n, c, f, t = spec.size()
            spec = spec.reshape(b*n, c, f, t)
            b, n, c, h, w = bboxes.size()
            bboxes = bboxes.reshape(b*n, c, h, w)
            bboxes = bboxes.squeeze(1)

            expanded_names = []
            for n in name:
                expanded_names.extend([n] * 5)
            name = expanded_names

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
        
        with torch.no_grad():
            heatmap_img_query, heatmap_aud = audio_visual_model(image.float(), spec.float())#heatmap_img_query是img_attn heatmap_aud是cross_attn
            obj_prior_feat = object_saliency_model(image) #OGL专用的外部支路 图像过一个resnet全局平均成B 1 7 7

        heatmap_aud = F.interpolate(heatmap_aud, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_aud = heatmap_aud.data.cpu().numpy()

        # Image target-query prior used by IQR.
        heatmap_img_query = F.interpolate(heatmap_img_query, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_img_query = heatmap_img_query.data.cpu().numpy()

        # External object-saliency prior used by OGL.
        heatmap_obj_prior = F.interpolate(obj_prior_feat, size=(224, 224), mode='bicubic', align_corners=False)
        heatmap_obj_prior = heatmap_obj_prior.data.cpu().numpy()

        bboxes = bboxes.data.cpu().numpy()
        #TODO:这种相加真的有意义吗？先norm再相加 不同 map 的绝对置信度尺度已经被人为抹掉了
        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_aud = utils.normalize_img(heatmap_aud[i, 0])
            pred_img_query = utils.normalize_img(heatmap_img_query[i, 0])
            pred_iqr = utils.normalize_img(pred_aud * args.alpha + pred_img_query * (1 - args.alpha)) #IQR就是aud结果和img自己查找加权
            pred_obj_prior = utils.normalize_img(heatmap_obj_prior[i, 0])
            pred_ogl = utils.normalize_img(pred_aud * args.alpha + pred_obj_prior * (1 - args.alpha)) #OGL就是aud结果和图像先验加权
            pred_extra_iqr_ogl = utils.normalize_img(pred_aud * args.alpha + pred_img_query * (1 - args.alpha) * 0.5 + pred_obj_prior * (1 - args.alpha) * 0.5) # extra就是三者加权0.6 0.2 0.2

            gt_map = bboxes[i]
            threshold = 0.6

            _, _, _, aud_infer_map = evaluator_aud.cal_CIOU(pred_aud, gt_map, name[i], threshold)
            _, _, _, img_query_infer_map = evaluator_img_query.cal_CIOU(pred_img_query, gt_map, name[i], threshold)
            _, _, _, iqr_infer_map = evaluator_iqr.cal_CIOU(pred_iqr, gt_map, name[i], threshold)
            _, _, _, obj_prior_infer_map = evaluator_obj_prior.cal_CIOU(pred_obj_prior, gt_map, name[i], threshold)
            _, _, _, ogl_infer_map = evaluator_ogl.cal_CIOU(pred_ogl, gt_map, name[i], threshold)
            _, _, _, extra_iqr_ogl_infer_map = evaluator_extra_iqr_ogl.cal_CIOU(pred_extra_iqr_ogl, gt_map, name[i], threshold)

            if epoch == -2: # drawing tool
                os.makedirs(os.path.join(viz_dir, name[i]), exist_ok=True)

                denorm_image = inverse_normalize(image[i]).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                denorm_image = (denorm_image*255).astype(np.uint8)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'image.jpg'), denorm_image)

                # visualize bboxes on raw images
                # gt_boxes_img = utils.visualize(denorm_image, gt_map) #, test_set=testset)
                overlay = np.zeros_like(denorm_image)
                overlay[gt_map == 1] = [0, 0, 255]
                gt_boxes_img = cv2.addWeighted(denorm_image, 1, overlay, 0.9, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'gt_boxes.jpg'), gt_boxes_img)

                # visualize predicted segmentation masks
                overlay = np.zeros_like(denorm_image)
                overlay[aud_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_aud.jpg'), highlighted_image)

                overlay = np.zeros_like(denorm_image)
                overlay[img_query_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_img_query.jpg'), highlighted_image)

                overlay = np.zeros_like(denorm_image)
                overlay[iqr_infer_map == 1] = [0, 255, 0]
                highlighted_image = cv2.addWeighted(gt_boxes_img, 1, overlay, 0.7, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'mask_iqr.jpg'), highlighted_image)

                # visualize heatmaps
                heatmap_pred_aud = np.uint8(pred_aud*255)
                heatmap_pred_aud = cv2.applyColorMap(heatmap_pred_aud[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_aud, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_aud.jpg'), fin)

                heatmap_pred_img = np.uint8(pred_img_query*255)
                heatmap_pred_img = cv2.applyColorMap(heatmap_pred_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_img, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_img_query.jpg'), fin)

                heatmap_pred_iqr = np.uint8(pred_iqr*255)
                heatmap_pred_iqr = cv2.applyColorMap(heatmap_pred_iqr[:, :, np.newaxis], cv2.COLORMAP_JET)
                fin = cv2.addWeighted(heatmap_pred_iqr, 0.6, np.uint8(denorm_image), 0.4, 0)
                cv2.imwrite(os.path.join(viz_dir, name[i], 'pred_iqr.jpg'), fin)

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc
    
    mAP_aud, _, auc_aud = compute_stats(evaluator_aud)
    mAP_img_query, _, auc_img_query = compute_stats(evaluator_img_query)
    mAP_iqr, _, auc_iqr = compute_stats(evaluator_iqr)
    mAP_obj_prior, _, auc_obj_prior = compute_stats(evaluator_obj_prior)
    mAP_ogl, _, auc_ogl = compute_stats(evaluator_ogl)
    mAP_extra_iqr_ogl, _, auc_extra_iqr_ogl = compute_stats(evaluator_extra_iqr_ogl)
    if epoch == -1:
        model_dir = os.path.join(args.model_dir, args.experiment_name)
        save_all_metrics(
            evaluator_aud,
            evaluator_img_query,
            evaluator_iqr,
            evaluator_obj_prior,
            evaluator_ogl,
            evaluator_extra_iqr_ogl,
            model_dir,
        )

    return mAP_aud, auc_aud, \
           mAP_img_query, auc_img_query, \
           mAP_iqr, auc_iqr, \
           mAP_obj_prior, auc_obj_prior, \
           mAP_ogl, auc_ogl, \
           mAP_extra_iqr_ogl, auc_extra_iqr_ogl

def save_sorted_metrics(evaluator, output_path):
    """Save sorted metrics cIoU for each file to a text file"""
    metrics = []
    for filename, ciou in evaluator.file_ciou.items():
        infer_ratio = evaluator.infer_ratio[filename]
        metrics.append((filename, ciou, infer_ratio))
    
    # Sort by cIoU in descending order
    metrics.sort(key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w') as f:
        f.write('Filename\tcIoU\tInfer Ratio\n')
        for filename, ciou, infer_ratio in metrics:
            f.write(f'{filename}\t{ciou:.4f}\t{infer_ratio:.4f}\n')

def save_all_metrics(evaluator_aud, evaluator_img_query, evaluator_iqr,
                     evaluator_obj_prior, evaluator_ogl,
                     evaluator_extra_iqr_ogl, viz_dir):
    """Save sorted metrics cIoU for each model's predictions"""
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save metrics for each model
    save_sorted_metrics(evaluator_aud, os.path.join(viz_dir, 'aud_metrics.txt'))
    save_sorted_metrics(evaluator_img_query, os.path.join(viz_dir, 'img_query_metrics.txt'))
    save_sorted_metrics(evaluator_iqr, os.path.join(viz_dir, 'iqr_metrics.txt'))
    save_sorted_metrics(evaluator_obj_prior, os.path.join(viz_dir, 'obj_prior_metrics.txt'))
    save_sorted_metrics(evaluator_ogl, os.path.join(viz_dir, 'ogl_metrics.txt'))
    save_sorted_metrics(evaluator_extra_iqr_ogl, os.path.join(viz_dir, 'extra_iqr_ogl_metrics.txt'))

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
