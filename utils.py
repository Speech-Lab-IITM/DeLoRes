import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_downstream_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--down_stream_task', default="iemocap", type=str,
                        help='''down_stream task name one of 
                        birdsong_freefield1010 , birdsong_warblr ,
                        speech_commands_v1 , speech_commands_v2
                        libri_100 , musical_instruments , iemocap , tut_urban , voxceleb1 , musan
                        ''')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size ')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default = False, type=str2bool,
                        help='number of total epochs to run')
    parser.add_argument('--pretrain_path', default=None, type=Path,
                        help='Path to Pretrain weights') 
    parser.add_argument('--freeze_effnet', default=True, type=str2bool,
                        help='Path to Pretrain weights')  
    parser.add_argument('--final_pooling_type', default='Avg', type=str,
                        help='valid final pooling types are Avg,Max')                                                            
    parser.add_argument('--load_only_efficientNet',default = True,type =str2bool)  
    parser.add_argument('--tag',default = "pretrain_big",type =str)
    parser.add_argument('--exp-dir',default='./exp/',type=Path,help="experiment root directory")    
    parser.add_argument('--lr',default=0.001,type=float,help="experiment root directory")                    
    return parser


def freeze_effnet(model):
    logger=logging.getLogger("__main__")
    logger.info("freezing effnet weights")
    for param in model.model_efficient.parameters():
        param.requires_grad = False

def load_pretrain(path,model,
                load_only_effnet=False,freeze_effnet=False):
    logger=logging.getLogger("__main__")
    logger.info("loading from checkpoint only weights : "+ str(path))
    checkpoint = torch.load(path)
    if load_only_effnet :
        for key in checkpoint['state_dict'].copy():
            if not 'model_efficient' in key:
                del checkpoint['state_dict'][key]
    mod_missing_keys,mod_unexpected_keys   = model.load_state_dict(checkpoint['state_dict'],strict=False)
    logger.info("Model missing keys")
    logger.info(mod_missing_keys)
    print(mod_missing_keys)
    logger.info("Model unexpected keys")
    logger.info(mod_unexpected_keys)
    print(mod_unexpected_keys)
    if freeze_effnet : 
        logger.info("freezing effnet weights")
        for param in model.model_efficient.parameters():
            param.requires_grad = False
    logger.info("done loading")
    return model

def resume_from_checkpoint(path,model,optimizer):
    logger = logging.getLogger("__main__")
    logger.info("loading from checkpoint : "+path)
    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']  
    logger.info("Task :: {}".format(checkpoint['down_stream_task']))
    mod_missing_keys,mod_unexpected_keys = model.load_state_dict(checkpoint['state_dict'],strict=False)  
    opt_missing_keys,opt_unexpected_keys = optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Model missing keys",mod_missing_keys)
    logger.info("Model unexpected keys",mod_unexpected_keys)
    logger.info("Opt missing keys",opt_missing_keys)
    logger.info("Opt unexpected keys",opt_unexpected_keys)
    logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    return start_epoch , model, optimizer


def save_to_checkpoint(down_stream_task,dir,epoch,model,opt):
    torch.save({
            'down_stream_task': down_stream_task,
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : opt.state_dict()
            },
            os.path.join('.',dir,'models', 'checkpoint_' + str(epoch) + "_" + '.pth.tar')
    )

def set_seed(seed = 31):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def move_to_gpu(*args):
    if torch.cuda.is_available():
        for item in args:
            item.cuda()

class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if isinstance(val, (torch.Tensor)):
            val = val.numpy()
            self.val = val
            self.sum += np.sum(val) 
            self.count += np.size(val)
        self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


#-------------------------------------------------------------------------------------#

def calc_norm_stats(train_dataset, test_dataset, n_stats=50000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.
    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """

    # def data_for_stats(data_src):
    #     # use all files for LOO-CV (Leave One Out CV)
    #     if data_src.loocv:
    #         return data_src
    #     # use training samples only for non-LOOCV (train/eval/test) split.
    #     return data_src.subset([0])

    # stats_data = data_for_stats(data_src)

    train_files = os.listdir(train_dataset.feat_root)
    test_files = os.listdir(test_dataset.feat_root)

    n_stats = min(n_stats, len(train_files + test_files))
    n_stats_train = int(n_stats * (len(train_files) / len(train_files + test_files)))
    n_stats_test = int(n_stats * (len(test_files) / len(train_files + test_files)))

    logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')

    sample_idxes_train = np.random.choice(range(len(train_files)), size=n_stats_train, replace=False)
    sample_idxes_test = np.random.choice(range(len(test_files)), size=n_stats_test, replace=False)

    X = [train_dataset[i] for i in tqdm(sample_idxes_train)] + [test_dataset[i] for i in tqdm(sample_idxes_test)]
    X = np.hstack(X)

    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats
