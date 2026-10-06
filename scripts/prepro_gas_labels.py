"""
Build the attribute vocabulary / scene vocabulary and the multi-label pseudo-targets used by GAS-Net
(paper Sec. 3.4.1 Steps 2-5 and Sec. 3.4.4).  Everything is derived from TRAINING captions only
(Karpathy 'train' + 'restval'); the rows of validation / test images in the label file are all zero.

Attribute vocabulary (Eq. 18-19):
    lemmatise + POS-tag captions -> keep adjectives and verb participles (JJ*, VBG, VBN)
    -> drop object words O -> drop words with count < tau_attr -> rank by TF-IDF -> top L
Scene vocabulary (Eq. 31-37):
    keep common nouns (NN/NNS) -> drop object words O -> drop words with count < tau_scene
    -> rank by TF-IDF -> keep the first K candidates whose WordNet hypernym chain contains one of
       {location, area, facility, structure, region, place}

Usage (MS COCO):
    python scripts/prepro_gas_labels.py --input_json data/dataset_coco.json \
        --output_h5 data/cocotalk_gas_label.h5 --output_json data/cocotalk_gas_vocab.json \
        --num_attrs 400 --tau_attr 5 --num_scenes 30 --tau_scene 10
Usage (Flickr30k):  --input_json data/dataset_flickr30k.json --tau_attr 3 --tau_scene 5 --num_attrs 300

The image order of the output equals the order of `images` in the Karpathy json, which is also the order
of cocotalk_attr.json / cocotalk_attr_label.h5 (see scripts/prepro_labels_attributes.py).

NLTK data needed once:  nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4');
                        nltk.download('averaged_perceptron_tagger')  (nltk>=3.9: 'averaged_perceptron_tagger_eng')
"""
from __future__ import print_function

import os
import io
import sys
import json
import math
import argparse
from collections import Counter, defaultdict

import numpy as np
import h5py

# Object vocabulary O: COCO category names (head words of multi-word names) + person words.
# 'orange' is deliberately NOT in the list: in captions it is overwhelmingly a colour (an attribute).
# Override with --object_vocab_file (one word per line) if you want a different O.
DEFAULT_OBJECTS = """
person bicycle car motorcycle airplane bus train truck boat light hydrant sign meter bench bird cat dog horse sheep
cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard ball kite bat glove
skateboard surfboard racket bottle glass cup fork knife spoon bowl banana apple sandwich broccoli carrot pizza donut
cake chair couch plant bed table toilet tv laptop mouse remote keyboard phone microwave oven toaster sink
refrigerator book clock vase scissors drier toothbrush
man woman people boy girl child kid guy lady men women player skier surfer
""".split()

# generic spatial nouns that satisfy the WordNet 'region/area/place' test but are not scenes.
# (extension to the paper; pass --scene_blacklist_file /dev/null to disable)
DEFAULT_SCENE_BLACKLIST = """
front side top back middle background foreground area place view corner edge center centre bottom distance
group row line part end space spot position region location structure facility surface
""".split()

SCENE_HYPERNYMS = set(['location', 'area', 'facility', 'structure', 'region', 'place'])
ATTR_TAGS = set(['JJ', 'JJR', 'JJS', 'VBG', 'VBN'])
ATTR_STOP = set(['be', 'have', 'do'])


def read_words(path):
    with io.open(path, encoding='utf8') as f:
        return set(w.strip() for w in f if w.strip())


def lemma_tag(tokens, lemmatizer, pos_tag):
    out = []
    for w, tag in pos_tag(tokens):
        if not w.isalpha():
            continue
        if tag.startswith('NN'):
            lemma = lemmatizer.lemmatize(w, 'n')
        elif tag.startswith('VB'):
            lemma = lemmatizer.lemmatize(w, 'v')
        elif tag.startswith('JJ'):
            lemma = lemmatizer.lemmatize(w, 'a')
        else:
            lemma = w
        out.append((lemma, tag))
    return out


def is_location_word(word, wn):
    """WordNet hypernym analysis (Eq. 33-34): any noun sense whose hypernym chain hits SCENE_HYPERNYMS."""
    for syn in wn.synsets(word, pos=wn.NOUN):
        for path in syn.hypernym_paths():
            for h in path:
                if SCENE_HYPERNYMS.intersection(h.lemma_names()):
                    return True
    return False


def tfidf_rank(tf, df, n_train, exclude, tau):
    cands = []
    for w, c in tf.items():
        if c < tau or w in exclude:
            continue
        cands.append((w, c * math.log(n_train / float(df[w]))))     # Eq. (18)/(32)
    cands.sort(key=lambda x: (-x[1], x[0]))
    return cands


def main(args):
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        nltk.pos_tag(['test'])
        wn.synsets('room')
    except (ImportError, LookupError) as e:
        sys.exit('NLTK / NLTK data problem: %s\nRun nltk.download for punkt, wordnet, omw-1.4, '
                 'averaged_perceptron_tagger(_eng).' % e)

    objects = read_words(args.object_vocab_file) if args.object_vocab_file else set(DEFAULT_OBJECTS)
    if args.scene_blacklist_file:
        blacklist = read_words(args.scene_blacklist_file)
    else:
        blacklist = set(DEFAULT_SCENE_BLACKLIST)

    imgs = json.load(open(args.input_json))['images']
    N = len(imgs)
    train_ix = [i for i, im in enumerate(imgs) if im['split'] in ('train', 'restval')]
    n_train = len(train_ix)
    print('%d images, %d training images (train + restval)' % (N, n_train))

    # ---- pass over TRAINING captions only -------------------------------------------------
    attr_tf, scene_tf = Counter(), Counter()
    attr_df, scene_df = defaultdict(int), defaultdict(int)
    img_attr, img_scene = {}, {}
    n_raw = Counter()
    for k, i in enumerate(train_ix):
        a_set, s_set = set(), set()
        for sent in imgs[i]['sentences']:
            for lemma, tag in lemma_tag(sent['tokens'], lemmatizer, nltk.pos_tag):
                n_raw[lemma] += 1
                if tag in ATTR_TAGS and lemma not in ATTR_STOP:
                    attr_tf[lemma] += 1
                    a_set.add(lemma)
                elif tag in ('NN', 'NNS'):
                    scene_tf[lemma] += 1
                    s_set.add(lemma)
        for w in a_set:
            attr_df[w] += 1
        for w in s_set:
            scene_df[w] += 1
        img_attr[i], img_scene[i] = a_set, s_set
        if (k + 1) % 20000 == 0:
            print('  tagged %d / %d training images' % (k + 1, n_train))

    # ---- attribute vocabulary ---------------------------------------------------------------
    after_obj = [w for w in attr_tf if w not in objects]
    ranked = tfidf_rank(attr_tf, attr_df, n_train, objects, args.tau_attr)
    attr_vocab = [w for w, _ in ranked[:args.num_attrs]]
    print('[attributes] raw tokens %d | after POS filter %d | after object removal %d | after freq filter %d | final L = %d'
          % (len(n_raw), len(attr_tf), len(after_obj), len(ranked), len(attr_vocab)))

    # ---- scene vocabulary -------------------------------------------------------------------
    after_obj_s = [w for w in scene_tf if w not in objects]
    ranked_s = tfidf_rank(scene_tf, scene_df, n_train, objects | blacklist, args.tau_scene)
    scene_vocab = []
    for w, _ in ranked_s:                               # walk down the TF-IDF ranking, keep validated words
        if is_location_word(w, wn):
            scene_vocab.append(w)
            if len(scene_vocab) == args.num_scenes:
                break
    print('[scenes]     raw tokens %d | after POS filter %d | after object removal %d | after freq filter %d | final K = %d'
          % (len(n_raw), len(scene_tf), len(after_obj_s), len(ranked_s), len(scene_vocab)))
    print('attribute vocab (first 40):', attr_vocab[:40])
    print('scene vocab:', scene_vocab)

    # ---- multi-hot pseudo-labels (Eq. 19, 37); val/test rows stay 0 --------------------------------
    a_idx = {w: k for k, w in enumerate(attr_vocab)}
    s_idx = {w: k for k, w in enumerate(scene_vocab)}
    A = np.zeros((N, len(attr_vocab)), dtype='uint8')
    S = np.zeros((N, len(scene_vocab)), dtype='uint8')
    for i in train_ix:
        for w in img_attr[i]:
            if w in a_idx:
                A[i, a_idx[w]] = 1
        for w in img_scene[i]:
            if w in s_idx:
                S[i, s_idx[w]] = 1
    print('avg #positive attributes / train image: %.2f | scenes: %.2f' % (A[train_ix].sum(1).mean(), S[train_ix].sum(1).mean()))

    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('attr_labels', data=A, compression='gzip')
        f.create_dataset('scene_labels', data=S, compression='gzip')
    json.dump({'attr_vocab': attr_vocab, 'scene_vocab': scene_vocab,
               'attr_tfidf': dict(ranked[:args.num_attrs]), 'args': vars(args)},
              open(args.output_json, 'w'), indent=1)
    print('wrote', args.output_h5, 'and', args.output_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default='data/dataset_coco.json', help='Karpathy split json (dataset_coco.json / dataset_flickr30k.json)')
    parser.add_argument('--output_h5', default='data/cocotalk_gas_label.h5')
    parser.add_argument('--output_json', default='data/cocotalk_gas_vocab.json')
    parser.add_argument('--num_attrs', type=int, default=400, help='L (COCO 400, Flickr30k 300)')
    parser.add_argument('--num_scenes', type=int, default=30, help='K')
    parser.add_argument('--tau_attr', type=int, default=5, help='min count, attributes (COCO 5, Flickr30k 3)')
    parser.add_argument('--tau_scene', type=int, default=10, help='min count, scenes (COCO 10, Flickr30k 5)')
    parser.add_argument('--object_vocab_file', default='', help='optional file with the object vocabulary O (one word per line)')
    parser.add_argument('--scene_blacklist_file', default='', help='optional replacement for the built-in generic-spatial-noun blacklist')
    main(parser.parse_args())
