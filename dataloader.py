from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))


        self.ix_to_word = self.info['ix_to_word']
        self.word_to_ix = self.info['word_to_ix']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)


        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')



        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        # [GAS-Net] region boxes (geometry branch) and pooled scene features (scene branch)
        self.input_box_dir = getattr(self.opt, 'input_box_dir', '')
        self.input_scene_dir = getattr(self.opt, 'input_scene_dir', '')
        self.use_box = bool(getattr(self.opt, 'use_geo', 1)) and len(self.input_box_dir) > 0
        self.use_scene_feat = bool(getattr(self.opt, 'use_scene', 1)) and len(self.input_scene_dir) > 0
        print('DataLoader boxes: ', self.input_box_dir if self.use_box else 'not used',
              '| scene feats: ', self.input_scene_dir if self.use_scene_feat else 'not used (falls back to fc feats)')

        # [GAS-Net] attribute / scene pseudo-labels, [num_images, L] and [num_images, K] (0/1).
        # Built from TRAINING captions only by scripts/prepro_gas_labels.py (rows of val/test images are 0).
        self.attr_labels = None
        self.scene_labels = None
        gas_h5 = getattr(self.opt, 'input_gas_label_h5', '')
        if len(gas_h5) > 0 and os.path.isfile(gas_h5):
            print('DataLoader loading GAS label file: ', gas_h5)
            with h5py.File(gas_h5, 'r') as f:
                self.attr_labels = f['attr_labels'][:].astype('float32')
                self.scene_labels = f['scene_labels'][:].astype('float32')
        elif len(gas_h5) > 0:
            print('WARNING: GAS label file not found: ', gas_h5)
        self.num_attrs = 0 if self.attr_labels is None else self.attr_labels.shape[1]
        self.num_scenes = 0 if self.scene_labels is None else self.scene_labels.shape[1]
        print('attribute vocab size L = %d, scene vocab size K = %d' % (self.num_attrs, self.num_scenes))


        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape  # [616767, 16]
        print("seq_size:{0}".format(seq_size))
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)




        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))
        if self.attr_labels is not None:
            assert self.attr_labels.shape[0] == self.num_images and self.scene_labels.shape[0] == self.num_images, \
                'GAS label file and caption label file must have the same image order/number'


        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)            
            else: # restval
                self.split_ix['train'].append(ix)


        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))


        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img
        fc_batch = [] 
        att_batch = []
        box_batch = []
        scene_batch = []
        ix_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')   # [BS, 18]
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')   # [BS, 18]
        wrapped = False
        infos = []
        gts = []
        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_box, tmp_scene, ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            box_batch.append(tmp_box)
            scene_batch.append(tmp_scene)
            ix_batch.append(ix)
            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)   # [BS, 18]
            if tmp_wrapped:
                wrapped = True
            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # every image-level input is repeated seq_per_img times *consecutively* (rows i*spi .. (i+1)*spi-1
        # belong to image i); the model relies on this layout (`rep`) to run the context branches once per image.
        def repeat_rows(lst):
            return np.stack([x for x in lst for _ in range(seq_per_img)])

        data = {}
        data['fc_feats'] = repeat_rows(fc_batch)
        max_att_len = max([_.shape[0] for _ in att_batch])
        # merge att_feats
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]], dtype='float32')    # [BS, 36, 2048]
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')     # [BS, 36]
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        # [GAS-Net] boxes [BS,36,4], scene features [BS,ds], attribute / scene labels [BS,L] / [BS,K]
        data['boxes'] = repeat_rows(box_batch) if box_batch[0] is not None else None
        data['scene_feats'] = repeat_rows(scene_batch) if scene_batch[0] is not None else None
        data['attr_labels'] = repeat_rows([self.attr_labels[i] for i in ix_batch]) if self.attr_labels is not None else None
        data['scene_labels'] = repeat_rows([self.scene_labels[i] for i in ix_batch]) if self.scene_labels is not None else None

        data['labels'] = np.vstack(label_batch)    # [BS, 18]
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))   # calculates the non-zero elements in each row of data['labels'] + 2(for start and end tokens)
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        # This function returns a tuple that is further passed to collate_fn
        ix = index
        img_id = str(self.info['images'][ix]['id'])
        # att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        att_feat = np.load(os.path.join(self.input_att_dir, img_id + '.npy'))
        # Reshape to K x C
        att_feat = att_feat.reshape(-1, att_feat.shape[-1])
        fc_feat = np.load(os.path.join(self.input_fc_dir, img_id + '.npy'))
        # [GAS-Net] region boxes (x1,y1,x2,y2), same order as the region features
        box = None
        if self.use_box:
            box = np.load(os.path.join(self.input_box_dir, img_id + '.npy')).astype('float32').reshape(-1, 4)
            assert box.shape[0] == att_feat.shape[0], 'number of boxes and region features differ for image %s' % img_id
        # [GAS-Net] pooled scene feature
        scene = None
        if self.use_scene_feat:
            scene = np.load(os.path.join(self.input_scene_dir, img_id + '.npy')).astype('float32').reshape(-1)
        return (fc_feat, att_feat, box, scene, ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = list(next(self.split_loader))   # (fc, att, box, scene, ix)

        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]
