# GAS-Net (Geometry / Attribute / Scene guided fusion) on top of the MAD+SAP baseline code

Drop these files over the baseline folder (or into a copy of it). `misc/util.py`, `cider-master/` and
`coco-caption/` are the same as for the baseline and are found relative to this folder
(or pass `--cider_path`, `--coco_caption_path`).

| file | what changed |
|---|---|
| `models/AttModel_GASNet.py` | **new** model: geometric / attribute / scene attention, CMI + TGA fusion, Up-Down decoder fed with `v_fusion` |
| `models/__init__.py` | registers `--caption_model gasnet` |
| `models/CaptionModel.py` | beam search untouched; only the `misc` import path is relative now |
| `dataloader.py` | also loads boxes, pooled scene features, attribute / scene labels |
| `opts.py` | paper hyper-parameters as defaults + GAS-Net / ablation flags |
| `train.py` | Stage 1 `L_XE + 0.1 L_attr + 1.0 L_scene`, Stage 2 SCST (`L_RL` only, lr 5e-5) |
| `eval.py`, `eval_utils.py` | pass boxes / scene feats, no hard-coded `/ghome` paths, `--model_id/--model_index` args |
| `scripts/prepro_gas_labels.py` | **new** attribute (L=400) + scene (K=30) vocab and pseudo-labels, training captions only |
| `scripts/extract_scene_feats.py` | **new** offline pooled scene features (ResNet-101 GAP) |

## Run
```bash
# 1. labels (needs nltk data: punkt, wordnet, omw-1.4, averaged_perceptron_tagger)
python scripts/prepro_gas_labels.py --input_json data/dataset_coco.json \
       --output_h5 data/cocotalk_gas_label.h5 --output_json data/cocotalk_gas_vocab.json
# 2. (optional but recommended) scene features
python scripts/extract_scene_feats.py --input_json data/cocotalk_attr.json --image_root <coco images> \
       --out_dir data/cocobu_scene_2048
# 3. train (add --input_scene_dir data/cocobu_scene_2048 --input_gas_label_h5 data/cocotalk_gas_label.h5 ...)
bash train.sh
# 4. test
python eval.py --model_id GASNet_coco --model_index 0016 --beam 1 --beam_size 3 --split test
```
Flickr30k: run the prepro with `--input_json dataset_flickr30k.json --tau_attr 3 --tau_scene 5 --num_attrs 300`
and point `--input_*` at the Flickr30k features.

## Ablation flags -> paper tables
| paper | flags |
|---|---|
| Baseline (Up-Down) | `--use_geo 0 --use_attr 0 --use_scene 0` |
| Table 8 geometry variants | `--geo_repr original\|signed`, `--geo_agg mean\|relation` |
| Table 9 single / pair branches | `--use_geo/--use_attr/--use_scene 0\|1` |
| Table 10 leave-one-out | one of `--use_geo/attr/scene 0`, or `--use_cmi 0`, or `--use_tga 0` |
| Table 11 fusion | direct sum `--use_cmi 0 --use_tga 0`; concat `--fusion_type concat`; CMI only `--use_tga 0`; TGA only `--use_cmi 0` |
| Table 12 top-T | `--top_t_attrs 5\|8\|10\|12\|15` |
| Table 13 beam | `--beam_size 1..7` at eval time |
