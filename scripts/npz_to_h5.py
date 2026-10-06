"""
Convert the .npz label file produced by prepro_gas_labels_heuristic.py (or by the real
prepro_gas_labels.py, which can also be pointed at an .npz if you prefer) into the actual
cocotalk_gas_label.h5 / flickr30k_gas_label.h5 that dataloader.py (--input_gas_label_h5) expects.

Needs h5py (`pip install h5py`) -- run this on a machine that has it, e.g. your training machine.

    python scripts/npz_to_h5.py --npz data/cocotalk_gas_label.npz --out data/cocotalk_gas_label.h5
"""
import argparse
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument('--npz', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

d = np.load(args.npz)
with h5py.File(args.out, 'w') as f:
    f.create_dataset('attr_labels', data=d['attr_labels'], compression='gzip')
    f.create_dataset('scene_labels', data=d['scene_labels'], compression='gzip')
print('wrote', args.out, 'attr_labels', d['attr_labels'].shape, 'scene_labels', d['scene_labels'].shape)
