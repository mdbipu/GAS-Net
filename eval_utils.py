from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(_HERE, 'misc'))
import util as utils    #import misc.utils as utils


def array_to_str(arr):
    out = []
    for i in range(len(arr)):
        out.append(str(arr[i]))
    return out


def language_eval(preds, model_id, split, eval_kwargs={}):
    import sys
    coco_caption_path = eval_kwargs.get('coco_caption_path', os.path.join(_HERE, 'coco-caption'))
    eval_results_dir = eval_kwargs.get('eval_results_dir', os.path.join(_HERE, 'eval_results'))
    sys.path.append(coco_caption_path)
    annFile = os.path.join(coco_caption_path, 'annotations', 'captions_val2014.json')
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir(eval_results_dir):
        os.makedirs(eval_results_dir)
    cache_path = os.path.join(eval_results_dir, model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()      # ids of all validation set images coco_val_2014 40504 as a list of int ids (Amr)


    # check if the preds image ids are found in the image ids of captions_val2014.json (Amr)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...



    # start evaluation (Amr)
    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()


    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out, imgToEval


def _to_cuda(arrs):
    return [_ if _ is None else torch.from_numpy(_).cuda() for _ in arrs]


def eval_split(model, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)   # [True]
    verbose_loss = eval_kwargs.get('verbose_loss', 1)   # [1]
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))   # [-1]
    split = eval_kwargs.get('split', 'test')   # ['val']
    lang_eval = eval_kwargs.get('language_eval', 1)   # [1]
    model_id = eval_kwargs.get('id','GASNet')
    beam = eval_kwargs.get('beam',0)   # [1]
    checkpoint_path = eval_kwargs.get('checkpoint_path','checkpoints')   # ['checkpoints']

    # Make sure that the model is in the evaluation mode
    model.eval()
    print('starting evaluation !')
    loader.reset_iterator(split)

    n = 0   # counts how many images have been brought till now for the specific split
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8      # counts how many times the loss have been calculated
    predictions = []

    if not os.path.isdir(os.path.join(checkpoint_path, 'logs')):
        os.makedirs(os.path.join(checkpoint_path, 'logs'))
    text_file = open(checkpoint_path+'/logs/cap_'+model_id+'.txt', "a")
    text_file.close()

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size   # counts how many images have been brought till now for the specific split

        # calculate the (word) loss (Amr)
        if data.get('labels', None) is not None and verbose_loss:
            tmp = [data['fc_feats'], data['labels'], data['masks'], data['att_feats'], data['boxes'], data['scene_feats']]
            fc_feats, labels, masks, att_feats, boxes, scene_feats = _to_cuda(tmp)

            loss = 0
            with torch.no_grad():
                # the model returns [num_gpu, 3] = (word, attr, scene); no labels are passed here, so the
                # validation loss is the plain word loss. `rep`: every image is repeated seq_per_img times.
                loss = model(fc_feats, att_feats, labels, masks, boxes=boxes, scene_feats=scene_feats,
                             rep=loader.seq_per_img)
                loss = loss[:, 0].mean()

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1


        # only leave one feature for each image, in case duplicate sample
        sel = np.arange(loader.batch_size) * loader.seq_per_img
        tmp = [data['fc_feats'][sel], data['att_feats'][sel],
               None if data['boxes'] is None else data['boxes'][sel],
               None if data['scene_feats'] is None else data['scene_feats'][sel]]
        fc_feats, att_feats, boxes, scene_feats = _to_cuda(tmp)


        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if(beam == 0):
                seq, _ = model(fc_feats, att_feats, boxes=boxes, scene_feats=scene_feats, mode='sample')
            else:
                seq = model(fc_feats, att_feats, boxes=boxes, scene_feats=scene_feats, mode='sample_beam')

        # converts seq from int tokens into words
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        seq = seq.cpu().numpy()


        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            print('image %s: %s' %(entry['image_id'], entry['caption']))
            predictions.append(entry)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))


        # finish the loop (Amr)
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
      

    # evaluate all of the generated captions
    lang_stats = None
    if lang_eval == 1:
        lang_stats, scores_each = language_eval(predictions, model_id, split, eval_kwargs)

        if verbose:
            text_file = open(checkpoint_path+'/logs/cap_'+model_id+'.txt', "a")
            for img_id in scores_each.keys():
                text_file.write('image %s, %s' % (scores_each[img_id]['image_id'], scores_each[img_id]['caption']))
                text_file.write('\n')
                text_file.write('cider {0:2f}'.format(scores_each[img_id]['CIDEr']))
                text_file.write('\n\n')
            text_file.close()

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
