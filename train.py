from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import os
from six.moves import cPickle
import opts as opts
import models
from dataloader import *
import eval_utils

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(_HERE, 'misc'))
import util as utils    #import misc.utils as utils

import traceback
import copy
import json
from collections import OrderedDict
import multiprocessing

CiderD_scorer = None

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(gen_result,greedy_res,data_gts):
    # Eq. (73): reward = CIDEr(sampled caption) - CIDEr(greedy caption)

    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = 5   #default value of MSCOCO
    
    # get greedy decoding baseline
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    




    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]
  

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    

    scores = cider_scores
    print('Cider scores(sample,argmax):', np.mean(scores[:batch_size]),np.mean(scores[batch_size:2*batch_size]))
    scores = scores[:batch_size] - scores[batch_size:2*batch_size]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def to_cuda(arrs):
    return [_ if _ is None else torch.from_numpy(_).cuda(device=0) for _ in arrs]


def train(opt):
    # setup dataloader
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size         # 9487
    opt.seq_length = loader.seq_length         # 16
    # [GAS-Net] sizes of the attribute (L) and scene (K) vocabularies come from the label file
    opt.num_attrs = loader.num_attrs           # 400 (COCO)
    opt.num_scenes = loader.num_scenes         # 30
    if opt.use_attr:
        assert loader.attr_labels is not None, 'use_attr=1 needs --input_gas_label_h5 (run scripts/prepro_gas_labels.py)'
    if opt.use_scene:
        assert loader.scene_labels is not None, 'use_scene=1 needs --input_gas_label_h5 (run scripts/prepro_gas_labels.py)'
    if opt.use_geo:
        assert loader.use_box, 'use_geo=1 needs --input_box_dir'
    
    #set the checkpoint path
    opt.checkpoint_path =  os.path.join(opt.checkpoint_path,opt.id)  #'checkpoints/GASNet_coco'
    isExists=os.path.exists(opt.checkpoint_path)
    if not isExists:
        os.makedirs(opt.checkpoint_path) 
        os.makedirs(opt.checkpoint_path+'/logs') 
        print(opt.checkpoint_path +' creating !')
    else:
        print(opt.checkpoint_path +' already exists!')
        if not os.path.exists(opt.checkpoint_path+'/logs'):
            os.makedirs(opt.checkpoint_path+'/logs')


    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open((os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(opt.start_from),'04') + '.pkl')),'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "att_feat_size", "rnn_size", "input_encoding_size",
                          "use_geo", "use_attr", "use_scene", "use_cmi", "use_tga", "fusion_size"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    best_val_score = None
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    #set up model, assure in training mode
    threshold = opt.threshold
    sc_flag = False
    num_gpu = opt.num_gpu
    
    model = models.setup(opt).cuda(device=0)
    model.train()
    print('number of trainable parameters: %.2f M' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    update_lr_flag = True
    dp_model = torch.nn.parallel.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')))
      
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob 
        
    optimizer.zero_grad()
    accumulate_iter = 0
    train_loss = 0
    
    while True:
        if update_lr_flag:

            # If start self critical training
            if sc_flag == False and opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                print('initializing CIDEr scorer...')
                s = time.time()
                global CiderD_scorer
                if (CiderD_scorer is None):
                    sys.path.insert(2, opt.cider_path)
                    from pyciderevalcap.ciderD.ciderD import CiderD
                    CiderD_scorer = CiderD(df=opt.cached_tokens)
                    # takes about 30s
                    print('initlizing CIDEr scorers in {:3f}s'.format(time.time() - s))
                sc_flag = True

            # Assign the learning rate
            if(sc_flag):
                if(epoch >= opt.self_critical_after and opt.self_critical_after != -1):
                    # [GAS-Net] SCST stage uses the smaller learning rate of the paper (5e-5)
                    opt.current_lr = opt.scst_learning_rate
            else:
                if epoch >= opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate

            for group in optimizer.param_groups:
                group['lr'] = opt.current_lr

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
    
            update_lr_flag = False  
    
        print('current_lr is {}'.format(opt.current_lr))
        start = time.time()
        data = loader.get_batch('train',opt.batch_size)
    
        torch.cuda.synchronize()
    
        tmp = [data['fc_feats'], data['labels'], data['masks'], data['att_feats'],
               data['boxes'], data['scene_feats'], data['attr_labels'], data['scene_labels']]
        fc_feats, labels, masks, att_feats, boxes, scene_feats, attr_labels, scene_labels = to_cuda(tmp)
        print('Read and process data:', time.time() - start)
        
        if not sc_flag:
            # ---- Stage 1: L_XE + lambda_a * L_attr + lambda_s * L_scene   (Eq. 71) ----
            out = dp_model(fc_feats, att_feats, labels, masks, boxes=boxes, scene_feats=scene_feats,
                           attr_labels=attr_labels, scene_labels=scene_labels, rep=opt.seq_per_img)
            out = out.mean(0)                    # [num_gpu, 3] -> (word, attr, scene)
            word_loss, attr_loss, scene_loss = out[0], out[1], out[2]
            total_loss = word_loss + opt.lambda_attr * attr_loss + opt.lambda_scene * scene_loss
            accumulate_iter = accumulate_iter + 1
            loss = total_loss /opt.accumulate_number
            loss.backward()
        else:
            # ---- Stage 2: SCST, objective = L_RL only (Eq. 72-73), no auxiliary / XE terms ----
            st = time.time()          
            sm = torch.zeros([num_gpu,1]).cuda(device=0) # indexs for sampling by probabilities
            gen_result,sample_logprobs = dp_model(fc_feats, att_feats, boxes=boxes, scene_feats=scene_feats,
                                                  sm=sm, rep=opt.seq_per_img, mode='sample')
            dp_model.eval()
            with torch.no_grad():
                greedy_res,_ = dp_model(fc_feats, att_feats, boxes=boxes, scene_feats=scene_feats,
                                        rep=opt.seq_per_img, mode='sample')
            dp_model.train()
            ed = time.time()
            print('GPU time is : {}s'.format(ed-st))
            reward = get_self_critical_reward(gen_result,greedy_res,data['gts'])
            word_loss = dp_model(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),mode='scst_forward')
            word_loss = word_loss.mean()
            total_loss = word_loss
            loss = word_loss
            loss.backward()             
            accumulate_iter = accumulate_iter + 1
    
        if accumulate_iter % opt.accumulate_number == 0:              
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item() * opt.accumulate_number
            end = time.time()

            if not sc_flag:
                print("iter {} (epoch {}), word_loss = {:.3f}, attr_loss = {:.3f}, scene_loss = {:.3f}, time/batch = {:.3f}".format(
                    iteration, epoch, word_loss.item(), attr_loss.item(), scene_loss.item(), end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}".format(iteration, epoch, np.mean(reward[:, 0]), end - start))

        torch.cuda.synchronize()
    
        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # make evaluation on validation set, and save model
        # (baseline: only when int(iteration/save_checkpoint_every) was in a hard-coded list [10,16,20,...];
        #  here: every `save_checkpoint_every` iterations)
        if (iteration % opt.save_checkpoint_every == 0) and (accumulate_iter % opt.accumulate_number == 0) and iteration > 0:
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json,
                           'num_images': -1,    # use all images in the specified split
                           'index_eval': 1,
                           'id': opt.id,
                           'beam': opt.beam,
                           'verbose_loss': 1,
                           'checkpoint_path': opt.checkpoint_path
                           }
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, loader, eval_kwargs)



            #save lang stats
            f_lang = open(opt.checkpoint_path+'/logs/lang_'+opt.id+'.txt','a')
            f_lang.write(str(iteration)+' '+str(iteration/opt.save_checkpoint_every)+'\n')
            f_lang.write('val loss '+str(val_loss)+'\n')
            if lang_stats:
                for key_lang in lang_stats:
                    f_lang.write(key_lang+' '+str(lang_stats[key_lang])+'\n')
            f_lang.close()
          



            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
    
            best_flag = False
            save_id = iteration/opt.save_checkpoint_every
            
            if best_val_score is None or current_score > best_val_score or current_score>threshold:

                best_val_score = current_score
                best_flag = True
                
                ##only save the improved models or when the CIDEr-D is larger than a given threshold
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                
                #record the lang stats for saved mdoel
                f_lang = open(opt.checkpoint_path+'/logs/Best_lang_'+opt.id+'.txt','a')
                f_lang.write(str(iteration)+' '+str(iteration/opt.save_checkpoint_every)+'\n')
                f_lang.write('val loss '+str(val_loss)+'\n')
                if lang_stats:
                    for key_lang in lang_stats:
                        f_lang.write(key_lang+' '+str(lang_stats[key_lang])+'\n')
                f_lang.close()                
    
            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            
            
            with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                cPickle.dump(infos, f)

    
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
    

opt = opts.parse_opt()
# os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
train(opt)
