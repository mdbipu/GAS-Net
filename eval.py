from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle
import opts as opts
import models
from dataloader import *
import eval_utils
import argparse
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(_HERE, 'misc'))
import util as utils    #import misc.utils as utils
import torch


# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
# (baseline: model_id / model_index were edited by hand inside this file; now they are arguments)
parser.add_argument('--model_id', type=str, default='GASNet_coco',
                help='the id (--id) of the trained model to evaluate')
parser.add_argument('--model_index', type=str, default='0016',
                help='checkpoint number, e.g. 0016 -> model<id>0016.pth')
parser.add_argument('--checkpoint_root', type=str, default='/gdata/mbipu/GASNet_coco/checkpoints',
                help='the --checkpoint_path used for training (checkpoints live in <root>/<model_id>/)')

#most frequently modifed options
parser.add_argument('--beam', type=int, default=1,
                help='whether beam search')
parser.add_argument('--beam_size', type=int, default=3,
                help='beam size (paper: 3). Was hard-coded in the baseline.')
parser.add_argument('--batch_size', type=int, default=40,
                help='if > 0 then overrule, otherwise load from checkpoint.')
#model information
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate (default: <checkpoint_root>/<model_id>/model<model_id><model_index>.pth)')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate (default: <checkpoint_root>/<model_id>/infos_<model_id><model_index>.pkl)')


# Basic options
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_fc_36',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_att_36',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_box_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_box_36',
                help='path to the boxes of the att feats')
parser.add_argument('--input_scene_dir', type=str, default='',
                help='pooled scene features (empty = fc feats fallback); must match what was used in training')
parser.add_argument('--input_gas_label_h5', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_gas_label.h5',
                help='attribute / scene label file (only needed to build the DataLoader; labels are not used at test time)')
parser.add_argument('--input_label_h5', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_attr_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_attr.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
# external code / outputs
parser.add_argument('--coco_caption_path', type=str, default=os.path.join(_HERE, 'coco-caption'))
parser.add_argument('--eval_results_dir', type=str, default=os.path.join(_HERE, 'eval_results'))
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_loss', type=int, default=1, 
                help='if we need to calculate loss.')
parser.add_argument('--verbose', type=int, default=1,
                help='if we need to print out all beam search beams.')
                      


opt = parser.parse_args()

model_id = opt.model_id
if len(opt.model) == 0:
    opt.model = os.path.join(opt.checkpoint_root, model_id, 'model' + model_id + opt.model_index + '.pth')
if len(opt.infos_path) == 0:
    opt.infos_path = os.path.join(opt.checkpoint_root, model_id, 'infos_' + model_id + opt.model_index + '.pkl')

# Load infos
with open((opt.infos_path), 'rb') as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

ignore = ["id", "batch_size", "start_from", "language_eval", 'model']
for k in vars(infos['opt']).keys():   
    if k != 'model':
        if k not in ignore:
            if k in vars(opt):
                #assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                pass
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

#modify
opt.drop_prob_lm = 0.0




# Setup the model  (all GAS-Net architecture options -- use_geo, num_attrs, fusion_size, ... -- come from infos['opt'])
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.beam_size = opt.beam_size
model.cuda()
model.eval()

loader = DataLoader(opt)
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
    print(lang_stats)



if not os.path.exists(opt.eval_results_dir):
    os.makedirs(opt.eval_results_dir)

with open(os.path.join(opt.eval_results_dir, 'res'+opt.id+'.txt'),"a") as text_file:
    text_file.write('{0}\n'.format(opt.model))
    text_file.write('beam={0} beam_size={1} split={2}\n'.format(opt.beam, opt.beam_size, opt.split))
    text_file.write('{0}\n'.format(lang_stats))


json.dump(split_predictions, open(os.path.join(opt.eval_results_dir, opt.id+'_results_RL.json'), 'w'))
