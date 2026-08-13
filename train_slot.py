import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import get_train_dataset, get_test_dataset, get_all_classes
from dataset_avs import get_ms3_dataset, get_s4_dataset
import random
import wandb

import test_model
from datetime import datetime, timedelta

import model_slot
import model_baseline
import utils
from training_history import render_training_curves, update_history

# from torch.utils.tensorboard import SummaryWriter   

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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='noname', help='experiment name (used for checkpointing and logging)')
    parser.add_argument('--model', default='jsa', choices=['jsa', 'av_mil'])

    # Data params
    parser.add_argument('--trainset', default='vggss', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='vggss', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--train_manifest_path', default='', type=str,
                        help='Optional text/CSV file listing training sample IDs')
    parser.add_argument('--train_metadata_path', default='', type=str,
                        help='Optional CSV file mapping training IDs to labels')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_manifest_path', default='', type=str,
                        help='Optional CSV file listing test sample IDs')
    parser.add_argument('--test_gt_path', default='', type=str)
    # parser.add_argument('--wandb', action='store_true')
    
    # hyper-params
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--aud_length', default=5.0, type=float)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regu")
    parser.add_argument("--lam1", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=1.0)
    parser.add_argument("--lam3", type=float, default=1.0)
    parser.add_argument("--infer_sharpening", type=float, default=0.1)
    parser.add_argument("--num_slots", type=int, default=3) #为啥是3
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--reciprocal_k", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=3, help="number of warmup epochs")
    parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'])
    # parser.add_argument("--scheduler", action='store_true', help='use scheduler or not')
    # parser.add_argument("--resume", action='store_true')

    # Distributed params
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=None)

    # Evaluation
    parser.add_argument('--alpha', default=0.6, type=float, help='alpha')
    parser.add_argument('--wandb',               type=str, default='false')
    parser.add_argument('--hard_img',            type=str, default='false')
    parser.add_argument('--hard_aud',            type=str, default='false')
    parser.add_argument('--rand_aud',            type=str, default='false')
    parser.add_argument("--scheduler",           type=str, default='false')
    parser.add_argument("--resume",              type=str, default='false')
    parser.add_argument('--save_visualizations', type=str, default='false')
    parser.add_argument('--eval_during_training', type=str, default='true',
                        help='Evaluate on the configured test set after every epoch')

    args = parser.parse_args()

    args.wandb = args.wandb in {'True', 'true'}
    args.hard_img = args.hard_img in {'True', 'true'}
    args.hard_aud = args.hard_aud in {'True', 'true'}
    args.rand_aud = args.rand_aud in {'True', 'true'}
    args.scheduler = args.scheduler in {'True', 'true'}
    args.resume = args.resume in {'True', 'true'}
    args.save_visualizations = args.save_visualizations in {'True', 'true'}
    args.eval_during_training = args.eval_during_training in {'True', 'true'}

    if args.experiment_name == 'noname':
        now = datetime.now()
        now = now.strftime('%m_%d_%H_%M_%S')
        args.experiment_name = "slotmse_%s_%s_%s" %(args.trainset, args.testset, now)
    return args


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
    setup_seed(args.seed)

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    if os.path.exists(model_dir):
        print('WARNING: Directory already exists.')
        # exit()

    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    if args.model == 'jsa':
        model = model_slot.mymodel(args)  #注意这里进的是mymodel 而不是slot
    else:
        model = model_baseline.AudioVisualMIL(args)
    print('Model loaded.')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    # Optimizer
    print('optimizer: {}, use scheduler: {}'.format(args.optimizer, args.scheduler))
    if args.optimizer == 'adam':
        optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)
    elif args.optimizer == 'sgd':
        optimizer, scheduler = utils.build_optimizer_and_scheduler_sgd(model, args)
    else:
        exit()

    scaler = torch.cuda.amp.GradScaler()

    print('train dataset: ', args.trainset)
    train_dataset = get_train_dataset(args, hard_img=args.hard_img, hard_aud=args.hard_aud, rand_aud=args.rand_aud)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
    print(f'train samples: {len(train_dataset)}, batches per epoch: {len(train_loader)}')

    test_loader = None
    object_saliency_model = None
    if args.eval_during_training:
        if args.testset == 'ms3':
            test_dataset = get_ms3_dataset(args)
        elif args.testset == 's4':
            test_dataset = get_s4_dataset(args)
        else:
            test_dataset = get_test_dataset(args, args.testset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False,
            persistent_workers=args.workers > 0)
        print(f'{args.testset} samples: {len(test_dataset)}')

        object_saliency_model = torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        object_saliency_model.avgpool = nn.Identity()
        object_saliency_model.fc = nn.Sequential(
            nn.Unflatten(1, (512, 7, 7)),
            NormReducer(dim=1),
            Unsqueeze(1)
        )
        object_saliency_model = object_saliency_model.cuda(args.gpu)

    wandbRun = wandb.init(project = 'SSL_JSA_%s' %(args.trainset),
                          config = vars(args),
                          name = args.experiment_name,
                          anonymous='allow',
                          mode= 'online' if args.wandb else 'disabled')
    
    start_epoch = 0
    best = [0.0] * 10
    run_start = time.time()
    history_path = os.path.join(model_dir, 'epoch_metrics.csv')
    curves_path = os.path.join(model_dir, 'training_curves.png')

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_metrics = train(train_loader, model, optimizer, scaler, epoch, args)

        if args.scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")

        lr_metrics = {
            'train/lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        }
        wandb.log(lr_metrics)

        eval_metrics = {}
        if args.eval_during_training:
            best, eval_metrics = validate(
                test_loader, args.testset, model, object_saliency_model,
                epoch, best, model_dir, args
            )
        # Checkpoint
        ckp = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1}
        torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
        print(f"Latest model saved to {model_dir}")

        history_record = {
            'epoch': epoch + 1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_seconds': time.time() - epoch_start,
            'train_total_loss': train_metrics['train/total_loss'],
            'train_info_loss': train_metrics['train/info_loss'],
            'train_recon_loss': train_metrics['train/recon_loss'],
            'train_div_loss': train_metrics['train/div_loss'],
            'train_attention_match_loss': train_metrics['train/att_loss'],
            'train_weighted_recon_loss': args.lam1 * train_metrics['train/recon_loss'],
            'train_weighted_div_loss': args.lam2 * train_metrics['train/div_loss'],
            'train_weighted_attention_match_loss': args.lam3 * train_metrics['train/att_loss'],
        }
        if eval_metrics:
            metric_prefixes = {
                'aud': 'AUD',
                'img_query': 'IMG_QUERY',
                'iqr': 'IQR',
                'obj_prior': 'OBJ_PRIOR',
                'ogl': 'OGL',
                'extra_iqr_ogl': 'EXTRA_IQR_OGL',
            }
            for history_prefix, eval_prefix in metric_prefixes.items():
                history_record[f'{history_prefix}_ciou'] = eval_metrics[
                    f'{eval_prefix}_{args.testset}/cIoU'
                ]
                history_record[f'{history_prefix}_auc'] = eval_metrics[
                    f'{eval_prefix}_{args.testset}/auc'
                ]
        try:
            update_history(history_path, history_record)
            render_training_curves(
                history_path, curves_path, title=args.experiment_name
            )
            print(
                f"Epoch history updated: {history_path}; curves: {curves_path}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"WARNING: Failed to update training curves: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        elapsed = time.time() - run_start
        completed_epochs = epoch - start_epoch + 1
        average_epoch_time = elapsed / completed_epochs
        remaining = average_epoch_time * (args.epochs - epoch - 1)
        finish_time = datetime.now() + timedelta(seconds=remaining)
        print(
            f"Epoch {epoch + 1}/{args.epochs} finished in "
            f"{timedelta(seconds=int(time.time() - epoch_start))}; "
            f"overall ETA {timedelta(seconds=int(remaining))}, "
            f"estimated finish {finish_time:%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    final_ckp = {'model': model.state_dict(), 'epoch': args.epochs}
    torch.save(final_ckp, os.path.join(model_dir, 'final.pth'))
    print(f"Final model saved to {model_dir}")
    total_elapsed = time.time() - run_start
    print(
        f"Total training time: {timedelta(seconds=int(total_elapsed))}",
        flush=True,
    )

    wandbRun.finish()
    print(args)
    return

def train(train_loader, model, optimizer, scaler, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    info_losses = AverageMeter('Info', ':.3f')
    con_losses = AverageMeter('Con', ':.3f')
    div_losses = AverageMeter('Div', ':.3f')
    att_losses = AverageMeter('Att', ':.3f')
    total_losses = AverageMeter('Total', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, info_losses, con_losses, div_losses, att_losses,
         total_losses],
        prefix="Warmup: [{}]".format(epoch) if epoch < args.warmup else "Train: [{}]".format(epoch),
    )

    end = time.time()
    processed_batches = 0
    for i, (frame, spec, bboxes, file_id, label) in enumerate(train_loader):
        processed_batches += 1
        batch_size = frame.size(0)
        data_time.update(time.time() - end)

        if args.gpu is not None:
            frame = frame.cuda(args.gpu, non_blocking=True) #b,3,224,224
            spec  = spec.cuda(args.gpu, non_blocking=True) #b,1,257,501

        with torch.cuda.amp.autocast():
            info_loss, recon_loss, div_loss, att_loss = model(frame.float(), spec.float())
            if epoch < args.warmup:
                loss = info_loss
            else:
                loss = info_loss + args.lam1 * recon_loss + args.lam2 * div_loss + args.lam3 * att_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        info_losses.update(info_loss.item(), batch_size)
        con_losses.update(recon_loss.item(), batch_size)
        div_losses.update(div_loss.item(), batch_size)
        att_losses.update(att_loss.item(), batch_size)
        total_losses.update(loss.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0 or i == len(train_loader) - 1:
            remaining_batches = (
                len(train_loader) - i - 1
                + (args.epochs - epoch - 1) * len(train_loader)
            )
            train_eta = timedelta(seconds=int(batch_time.avg * remaining_batches))
            progress.display(i, suffix=f"train ETA {train_eta}")

    if processed_batches != len(train_loader):
        raise RuntimeError(
            f"Training loop consumed {processed_batches}/{len(train_loader)} batches"
        )
    
    metrics = {
        'train/info_loss': info_losses.avg,
        'train/recon_loss': con_losses.avg,
        'train/div_loss': div_losses.avg,
        'train/att_loss': att_losses.avg,
        'train/total_loss': total_losses.avg,
        'epoch': epoch
    }
    wandb.log(metrics)
    return metrics

def validate(test_loader, testset, model, object_saliency_model, epoch, best, model_dir,args):
    model.eval()
    object_saliency_model.eval()
    
    mAP_aud, auc_aud, \
    mAP_img_query, auc_img_query, \
    mAP_iqr, auc_iqr, \
    mAP_obj_prior, auc_obj_prior, \
    mAP_ogl, auc_ogl, \
    mAP_extra_iqr_ogl, auc_extra_iqr_ogl = \
        test_model.validate_img_aud(test_loader, model, object_saliency_model, './%s/%s/%s' %('final', args.trainset, testset), testset, epoch, args)
    
    if mAP_iqr > best[4]:
        ckp = {'model': model.state_dict(),
            'epoch': epoch+1,
            'selection_metric': 'IQR_cIoU',
            'selection_score': mAP_iqr}
        torch.save(ckp, os.path.join(model_dir, '%s_best.pth' %(testset)))
        print(
            f"Best model saved to {model_dir} "
            f"(IQR cIoU={mAP_iqr:.4f}, epoch={epoch + 1})"
        )

    best[0] = max(mAP_aud, best[0])
    best[1] = max(auc_aud, best[1])
    best[2] = max(mAP_img_query, best[2])
    best[3] = max(auc_img_query, best[3])
    best[4] = max(mAP_iqr, best[4])
    best[5] = max(auc_iqr, best[5])
    best[6] = max(mAP_ogl, best[6])
    best[7] = max(auc_ogl, best[7])
    best[8] = max(mAP_extra_iqr_ogl, best[8])
    best[9] = max(auc_extra_iqr_ogl, best[9])

    # Just for logging
    print('AUD_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_aud:.4f}', f'{auc_aud:.4f}', f'{best[0]:.4f}', f'{best[1]:.4f}')
    print('IMG_QUERY_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_img_query:.4f}', f'{auc_img_query:.4f}', f'{best[2]:.4f}', f'{best[3]:.4f}')
    print('IQR_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_iqr:.4f}', f'{auc_iqr:.4f}', f'{best[4]:.4f}', f'{best[5]:.4f}')
    print('OBJ_PRIOR_%s/cIoU, auc' %(testset), f'{mAP_obj_prior:.4f}', f'{auc_obj_prior:.4f}')
    print('OGL_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_ogl:.4f}', f'{auc_ogl:.4f}', f'{best[6]:.4f}', f'{best[7]:.4f}')
    print('EXTRA_IQR_OGL_%s/cIoU, auc, best_cIoU, best_auc' %(testset), f'{mAP_extra_iqr_ogl:.4f}', f'{auc_extra_iqr_ogl:.4f}', f'{best[8]:.4f}', f'{best[9]:.4f}')

    metrics = {
        'AUD_%s/cIoU' %(testset): mAP_aud,
        'AUD_%s/auc' %(testset): auc_aud,
        'AUD_%s/best_cIoU' %(testset): best[0],
        'AUD_%s/best_auc' %(testset): best[1],
        
        'IMG_QUERY_%s/cIoU' %(testset): mAP_img_query,
        'IMG_QUERY_%s/auc' %(testset): auc_img_query,
        'IMG_QUERY_%s/best_cIoU' %(testset): best[2],
        'IMG_QUERY_%s/best_auc' %(testset): best[3],

        'IQR_%s/cIoU' %(testset): mAP_iqr,
        'IQR_%s/auc' %(testset): auc_iqr,
        'IQR_%s/best_cIoU' %(testset): best[4],
        'IQR_%s/best_auc' %(testset): best[5],

        'OBJ_PRIOR_%s/cIoU' %(testset): mAP_obj_prior,
        'OBJ_PRIOR_%s/auc' %(testset): auc_obj_prior,

        'OGL_%s/cIoU' %(testset): mAP_ogl,
        'OGL_%s/auc' %(testset): auc_ogl,
        'OGL_%s/best_cIoU' %(testset): best[6],
        'OGL_%s/best_auc' %(testset): best[7],

        'EXTRA_IQR_OGL_%s/cIoU' %(testset): mAP_extra_iqr_ogl,
        'EXTRA_IQR_OGL_%s/auc' %(testset): auc_extra_iqr_ogl,
        'EXTRA_IQR_OGL_%s/best_cIoU' %(testset): best[8],
        'EXTRA_IQR_OGL_%s/best_auc' %(testset): best[9],
        'epoch': epoch
    }
    wandb.log(metrics)
    return best, metrics

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch, suffix=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if suffix:
            entries.append(suffix)
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main(args)
