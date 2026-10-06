import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))


def parse_opt():
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument('--beam', type=int, default=0,
                    help='whether use beam search in validation')
    parser.add_argument('--beam_size', type=int, default=3,
                    help='beam size used when beam search is on (paper: B=3). It was hard-coded in the baseline.')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--num_gpu', type=int, default=1,
                    help='num of gpus')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu_id')
    parser.add_argument('--threshold', type=float, default=1.05,
                    help='threshold of saving checkpoints')

    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-new-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--self_critical_after', type=int, default=50,
                    help='After what epoch do we start SCST? (-1 = disable; never finetune, 0 = finetune from start). Paper: 50 XE epochs')
    parser.add_argument('--train_split', type=str, default='train',
                        help='which split used to train')
    # Data directories
    parser.add_argument('--input_json', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_attr.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_fc_36',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_att_36',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='/gdata/mbipu/shareddata/cocobu_box_36',
                    help='[GAS-Net] path to the directory containing the boxes (x1,y1,x2,y2) of att feats, one <image_id>.npy [36,4] per image')
    parser.add_argument('--input_label_h5', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_attr_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--input_gas_label_h5', type=str, default='/gdata/mbipu/shareddata/data/cocotalk_gas_label.h5',
                    help='[GAS-Net] h5 with attribute / scene pseudo-labels, written by scripts/prepro_gas_labels.py')
    parser.add_argument('--input_scene_dir', type=str, default='',
                    help='[GAS-Net] directory with one pooled scene feature <image_id>.npy [scene_feat_size] per image '
                         '(scripts/extract_scene_feats.py). Empty = fall back to the mean-pooled region feature (fc feats).')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="gasnet",
                    help='gasnet')
    parser.add_argument('--id', type=str, default='GASNet_coco',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--rnn_size', type=int, default=1024,
                    help='size of the rnn in number of hidden nodes in each layer (paper: d_h = 1024)')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')

    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary (paper: 512)')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the decoder attention MLP')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')

    # [GAS-Net] context branches / fusion
    parser.add_argument('--use_geo', type=int, default=1, help='geometric attention branch (0/1)')
    parser.add_argument('--use_attr', type=int, default=1, help='attribute attention branch (0/1)')
    parser.add_argument('--use_scene', type=int, default=1, help='scene attention branch (0/1)')
    parser.add_argument('--use_cmi', type=int, default=1, help='cross-modal interaction z_int in the fusion (0/1)')
    parser.add_argument('--use_tga', type=int, default=1, help='triple gating attention in the fusion (0/1)')
    parser.add_argument('--fusion_type', type=str, default='gas', choices=['gas', 'concat'],
                    help="'gas' = sum of (gated) projections + CMI (Eq. 62); 'concat' = concatenation + linear (Table 11)")
    parser.add_argument('--geo_repr', type=str, default='signed', choices=['signed', 'original'],
                    help='pairwise geometry: signed offsets (Eq. 5) or |offsets| (Eq. 4)')
    parser.add_argument('--geo_agg', type=str, default='relation', choices=['relation', 'mean'],
                    help='geometric context aggregation: relation-aware (Eq. 8-10) or mean (Eq. 7)')
    parser.add_argument('--geo_dim', type=int, default=512, help='d_g, geometric embedding dimension')
    parser.add_argument('--geo_eps', type=float, default=1e-4, help='epsilon of Eq. 4/5')
    parser.add_argument('--geo_min_size', type=float, default=1.0,
                    help='boxes are in pixels; w,h are clamped to this minimum to avoid exploding ratios '
                         '(set to 0 if you feed normalised boxes)')
    parser.add_argument('--gas_att_size', type=int, default=512, help='d_a, attention space of the 3 branches')
    parser.add_argument('--attr_embed_size', type=int, default=512, help='d_e, attribute embedding size')
    parser.add_argument('--fusion_size', type=int, default=512, help='d_f, shared fusion space')
    parser.add_argument('--scene_feat_size', type=int, default=2048, help='d_s, pooled scene feature size')
    parser.add_argument('--top_t_attrs', type=int, default=10, help='T, number of predicted attributes that are kept (Eq. 22)')
    parser.add_argument('--lambda_attr', type=float, default=0.1, help='lambda_a of Eq. 71')
    parser.add_argument('--lambda_scene', type=float, default=1.0, help='lambda_s of Eq. 71')
    # (num_attrs / num_scenes are read from the label file by train.py)

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=80,
                    help='number of epochs (paper: 50 XE + 30 SCST = 80)')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size (paper: 32)')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='XE learning rate (paper: 5e-4)')
    parser.add_argument('--scst_learning_rate', type=float, default=5e-5,
                    help='SCST learning rate (paper: 5e-5)')

    parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=5,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay used for adam')
    parser.add_argument('--accumulate_number', type=int, default=1,
                        help='how many times it should accumulate the gradients, the truth batch_size=accumulate_number*batch_size')

    parser.add_argument('--scheduled_sampling_start', type=int, default=0,
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                    help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=-1,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=3540,
                    help='validate + save every N iterations (3540 = one epoch of 113,287 images at batch 32). '
                         'The baseline only validated at hard-coded multiples [10,16,20,...] of this value.')
    parser.add_argument('--checkpoint_path', type=str, default='/gdata/mbipu/GASNet_coco/checkpoints',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=10,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')

    # External code (previously hard-coded to /ghome/mbipu/Baseline_coco/...)
    parser.add_argument('--coco_caption_path', type=str, default=os.path.join(_HERE, 'coco-caption'),
                    help='path of the coco-caption repo (must contain annotations/captions_val2014.json)')
    parser.add_argument('--cider_path', type=str, default=os.path.join(_HERE, 'cider-master'),
                    help='path of the cider repo (pyciderevalcap)')
    parser.add_argument('--eval_results_dir', type=str, default=os.path.join(_HERE, 'eval_results'),
                    help='where evaluation caches / results are written')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"

    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    for k in ['use_geo', 'use_attr', 'use_scene', 'use_cmi', 'use_tga']:
        assert getattr(args, k) in (0, 1), "%s should be 0 or 1" % k
    assert args.top_t_attrs > 0, "top_t_attrs should be greater than 0"

    return args
