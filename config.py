"""
Adithya Bhaskar, 2022.
This file contains the configuration parameters that should be modified to change any of the aspects
of the model or its training.
"""
DATASET_DIR = "data/MS-SNSD-dataset-30"
SCALE_FACTOR = 10               # librosa uses the Mel spectrogram to approximate
                                # the STFT magnitude. The conversion of Mel to STFT
                                # is not lossless, and results in clipping effects
                                # for noisy audio.
                                # To reduce this, we multiply noisy audio file inputs
                                # by SCALE_FACTOR throughout.
MASK_MIN_VALUE = 0.05           # The minimum value to use for the mask
TARGET = 1                      # 0-1 range of clean-ness - we did not try anything less than 1
BATCH_SIZE = 1                  # Input audio lengths are different. Further, stoi computation
                                # is not parallelizable in the way we currently use it.
                                # Using a different BATCH_SIZE is thus not feasible - 
                                # we must stick to 1.
CHECKPT_DIR = "checkpoints/"    # The directory with checkpoints
NUM_GAN_EPOCHS = 8              # Original paper uses 200 -- but we have use so much more data
                                # for training that each GAN epoch took ~45 min / discriminator
                                # sub-epoch and ~52 min per generator sub-epoch, and ~30 min to
                                # save new files for the next epoch, leading to around
                                # 224 minutes for each GAN epoch. 8 seemed like a reasonable number.
NUM_DISCRIMINATOR_EPOCHS = 2    # Number of discriminator epochs in each GAN epoch.
                                # Please note that we used 1 for the first three epochs, and
                                # then 2 for the next 5 epochs. Since the resumption on Colab was
                                # done manually after epoch 2 (0-indexed and 3, 1-indexed),
                                # we did not need any extra code to do that.
NUM_GENERATOR_EPOCHS = 2        # Number of generator epochs in each GAN epoch
FORCE_RESTART = False           # Restart training from epoch 0
RESUME_FROM = 8                 # Epoch number (0-indexed) to resume from if FORCE_RESTART is False
CONTINUE = False                # Load epoch number but not model states -- was useful if on Colab
                                # you want to increase the number of epochs *after* training completes
save = True                     # Save model checkpoints during training
REPLACE = True                  # The npairs_8.pkl file has file-paths saved that are on google
                                # drive and thus not accessible from here. Setting replace will take
                                # the path a/b/c/.../x.ext in n_pairs and replace it by
                                # {CHECKPT_DIR}/x.ext