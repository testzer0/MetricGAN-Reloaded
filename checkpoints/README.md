## Note

Note that the epochX directory uses 0-indexing, while the checkpoints use 1-indexing.
This is somewhat unfortunate, but we decided to not change it so as to not use progress
(the new_pairs array does not depend on whether we use 0- or 1- indexing, since it is
saved across epochs -- hence thankfully this does not affect correctness).

The files in `epoch7/` are the cleaned versions of those in `data/MS-SNSD-dataset-30/train/noisy`
produced by our model after epoch 8 (1-indexed 8).

We did not include previous checkpoints in this repository to save space.