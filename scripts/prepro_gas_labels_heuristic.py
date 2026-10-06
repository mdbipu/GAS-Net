"""
APPROXIMATE stand-in for scripts/prepro_gas_labels.py, used only because this environment has no
internet access and no nltk / h5py installed.

It replaces:
  - NLTK's trained POS tagger              -> a hand-written suffix/lexicon heuristic
  - NLTK's WordNetLemmatizer                -> a hand-written suffix stemmer
  - WordNet hypernym search (Eq. 33-34)     -> membership in a curated scene-word lexicon
  - h5py                                    -> numpy .npz (convert to .h5 with the 3-line
                                                script printed at the end / scripts/npz_to_h5.py)

Everything else (TF-IDF ranking Eq. 18/32, frequency thresholds tau_attr/tau_scene, train-only
label construction Eq. 19/37, top-L / top-K selection) follows scripts/prepro_gas_labels.py exactly.

This is a lower-fidelity substitute for the paper's method (no trained tagger, no WordNet
reasoning) -- use scripts/prepro_gas_labels.py with real nltk locally for the reported numbers.
"""
from __future__ import print_function
import os
import sys
import json
import math
import argparse
from collections import Counter, defaultdict
import numpy as np

DEFAULT_OBJECTS = set("""
person bicycle car motorcycle airplane bus train truck boat light hydrant sign meter bench bird cat dog horse sheep
cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard ball kite bat glove
skateboard surfboard racket bottle glass cup fork knife spoon bowl banana apple sandwich broccoli carrot pizza donut
cake chair couch plant bed table toilet tv laptop mouse remote keyboard phone microwave oven toaster sink
refrigerator book clock vase scissors drier toothbrush hat shirt pants shoe shoes jacket helmet
man woman people boy girl child kid guy lady men women player skier surfer baby child boys girls
""".split())

STOPWORDS = set("""
a an the this that these those there here it its his her their our your my
is are was were be been being am do does did have has had having will would can could shall should may might must
i you he she we they him them me us
and or but if then than so because while although though when where which who whom whose what
of in on at by for with about against between into through during before after above below to from up down out off over under
again further once s t just very too also not no nor only own same
one two three four five six seven eight nine ten
""".split())

# participles ending in -ing that are actually common nouns, not actions/attributes
NOUN_ING = set("""
building wedding morning evening ceiling clothing painting ring king thing something nothing everything anything
wing string spring shopping bedding lighting seating flooring railing siding parking landing
""".split())

ADJ_LEXICON = set("""
red blue green yellow orange black white brown pink purple gray grey tan gold silver
wooden wood metal metallic plastic glass leather stone brick concrete
small large big tiny huge long short tall wide narrow thin thick heavy light
old new young empty full open closed dirty clean wet dry hot cold warm bright dark
round square flat curved striped spotted colorful plain fancy modern antique rustic
happy sad angry excited tired busy quiet loud calm
""".split())

ADJ_SUFFIXES = ('ful', 'ous', 'ive', 'able', 'ible', 'ish', 'less')

SCENE_LEXICON = set("""
beach park kitchen street airport restaurant farm forest room field bathroom bedroom garden yard pool
court station market mountain hill lake river ocean sea sky road bridge city town village house building
tower church stadium gym school classroom library museum zoo harbor dock platform runway track rink arena
plaza square alley sidewalk pavement driveway patio porch balcony rooftop hallway corridor studio workshop
garage barn stable pasture meadow desert jungle cave valley cliff shore coast island countryside wilderness
parking playground courtyard terrace lobby entrance doorway window shop store mall cafe cafeteria office
neighborhood skyline landscape waterfront pier boardwalk trail path sidewalk intersection crossing lane
""".split())


def lemma(w):
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y'
    if (w.endswith('es') and len(w) > 4 and (w[-3] in 'sxz' or w.endswith(('ches', 'shes')))):
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss') and len(w) > 3:
        return w[:-1]
    if w.endswith('ing') and len(w) > 5:
        stem = w[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            stem = stem[:-1]
        return stem
    if w.endswith('ed') and len(w) > 4:
        stem = w[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in 'aeiou':
            stem = stem[:-1]
        return stem
    return w


def classify(tok, objects):
    if not tok.isalpha() or len(tok) < 2 or tok in STOPWORDS:
        return None, None
    if tok in objects:
        return 'obj', tok
    if tok in ADJ_LEXICON:
        return 'attr', tok
    if tok.endswith(('ing', 'ed')) and len(tok) > 4 and tok not in NOUN_ING:
        return 'attr', lemma(tok)
    if tok.endswith(ADJ_SUFFIXES) and len(tok) > 4:
        return 'attr', tok
    return 'noun', lemma(tok)


def tfidf_rank(tf, df, n_train, exclude, tau):
    cands = []
    for w, c in tf.items():
        if c < tau or w in exclude:
            continue
        cands.append((w, c * math.log(n_train / float(df[w]))))
    cands.sort(key=lambda x: (-x[1], x[0]))
    return cands


def main(args):
    imgs = json.load(open(args.input_json))['images']
    N = len(imgs)
    train_ix = [i for i, im in enumerate(imgs) if im['split'] in ('train', 'restval')]
    n_train = len(train_ix)
    print('%d images total, %d training (train+restval)' % (N, n_train))

    attr_tf, scene_tf = Counter(), Counter()
    attr_df, scene_df = defaultdict(int), defaultdict(int)
    img_attr, img_scene = {}, {}
    n_raw = 0
    for k, i in enumerate(train_ix):
        a_set, s_set = set(), set()
        for sent in imgs[i]['sentences']:
            for tok in sent['tokens']:
                tok = tok.lower()
                n_raw += 1
                kind, lem = classify(tok, DEFAULT_OBJECTS)
                if kind == 'attr':
                    attr_tf[lem] += 1
                    a_set.add(lem)
                elif kind == 'noun':
                    scene_tf[lem] += 1
                    s_set.add(lem)
        for w in a_set:
            attr_df[w] += 1
        for w in s_set:
            scene_df[w] += 1
        img_attr[i], img_scene[i] = a_set, s_set
        if (k + 1) % 20000 == 0:
            print('  processed %d / %d training images' % (k + 1, n_train))

    ranked_a = tfidf_rank(attr_tf, attr_df, n_train, DEFAULT_OBJECTS, args.tau_attr)
    attr_vocab = [w for w, _ in ranked_a[:args.num_attrs]]

    ranked_s_all = tfidf_rank(scene_tf, scene_df, n_train, DEFAULT_OBJECTS, args.tau_scene)
    scene_vocab = [w for w, _ in ranked_s_all if w in SCENE_LEXICON][:args.num_scenes]

    print('[attributes] raw tokens=%d candidates(after freq)=%d final L=%d' % (n_raw, len(ranked_a), len(attr_vocab)))
    print('[scenes]     candidates(after freq)=%d in-lexicon=%d final K=%d' % (
        len(ranked_s_all), sum(1 for w, _ in ranked_s_all if w in SCENE_LEXICON), len(scene_vocab)))
    print('attribute vocab:', attr_vocab)
    print('scene vocab:', scene_vocab)

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
    print('avg positive attrs/train img: %.2f | scenes/train img: %.2f' % (A[train_ix].sum(1).mean(), S[train_ix].sum(1).mean()))
    print('val/test rows are all-zero (leak check): attr=%d scene=%d (should be 0)' % (
        A[[i for i in range(N) if i not in train_ix]].sum(), S[[i for i in range(N) if i not in train_ix]].sum()))

    np.savez_compressed(args.output_npz, attr_labels=A, scene_labels=S)
    json.dump({'attr_vocab': attr_vocab, 'scene_vocab': scene_vocab, 'args': vars(args),
               'note': 'heuristic fallback (no nltk/wordnet) -- see module docstring'},
              open(args.output_json, 'w'), indent=1)
    print('wrote', args.output_npz, 'and', args.output_json)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_json', required=True)
    p.add_argument('--output_npz', required=True)
    p.add_argument('--output_json', required=True)
    p.add_argument('--num_attrs', type=int, default=400)
    p.add_argument('--num_scenes', type=int, default=30)
    p.add_argument('--tau_attr', type=int, default=5)
    p.add_argument('--tau_scene', type=int, default=10)
    main(p.parse_args())
