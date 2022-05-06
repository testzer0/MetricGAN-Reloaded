"""
Adithya Bhaskar, 2022.
This file contains the classes and functions that define and train the model.
The usage of the model is in model/utility.py
"""

from config import *
from utils.globals import *
from utils.data import get_path_for_generator, get_generator_sample, \
    get_discriminator_sample, spectrogram_to_wav
import time
import datetime
import os
import re
import librosa
import soundfile as sf
import numpy as np
import math
import pickle
from pystoi import stoi

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import TensorDataset, DataLoader, Dataset, RandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


class Generator(nn.Module):
    def __init__(self, in_dim=257, out_dim=200):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lstms = nn.LSTM(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(in_features=2*self.out_dim, out_features=300)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.05)
        self.linear2 = nn.Linear(in_features=300, out_features=257)
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        lstm_out, _ = self.lstms(x)
        layer1_out = self.dropout(self.leaky_relu(self.linear1(lstm_out)))
        layer2_out = self.sigmoid(self.linear2(layer1_out))
        return layer2_out
    
class Discriminator(nn.Module):
    def __init__(self, in_dim=257):
        super().__init__()
        # We are passed (batch_size, 2, n_frames, in_dim=257) as input -> 
        # since we need both clean and noisy
        # Note that there is no 'channels_last' feature in pytorch
        self.in_dim = in_dim
        self.batch_norm = nn.BatchNorm2d(num_features=2)
        self.conv2d_sn1 = spectral_norm(nn.Conv2d(in_channels=2, \
            out_channels=15, kernel_size=(5,5), padding='valid'))
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2d_sn2 = spectral_norm(nn.Conv2d(in_channels=15, \
            out_channels=35, kernel_size=(7,7), padding='valid'))
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv2d_sn3 = spectral_norm(nn.Conv2d(in_channels=35, \
            out_channels=65, kernel_size=(9,9), padding='valid'))
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv2d_sn4 = spectral_norm(nn.Conv2d(in_channels=65, \
            out_channels=90, kernel_size=(11,11), padding='valid'))
        self.leaky_relu4 = nn.LeakyReLU()
        # pytorch has no global average pooling layer (i.e. (channels, h, w) -> channels)
        # use AdaptiveAvgPool2d to get (channels, 1, 1) then flatter
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()        
        # Now output should be batch_size x 50
        self.linear1 = spectral_norm(nn.Linear(in_features=90, \
            out_features=50))
        self.leaky_relu5 = nn.LeakyReLU()
        self.linear2 = spectral_norm(nn.Linear(in_features=50, \
            out_features=10))
        self.leaky_relu6 = nn.LeakyReLU()
        self.linear3 = spectral_norm(nn.Linear(in_features=10, \
            out_features=1))
        self.sigmoid = nn.Sigmoid()
        self.std = 0.1

    def std_step(self):
        """
        As it stands, we do not add Guassian noise to inputs, but we did
        experiment with it. When we did, the std was gradually decreased
        over the epochs.
        """
        self.std = self.std * 0.9

    def forward(self, x):
        x_normalized = self.batch_norm(x)
        # x_normalized = x_normalized + \
        #   (self.std**0.5)*torch.randn(x_normalized.shape).to(device)
        conv1_out = self.leaky_relu1(self.conv2d_sn1(x_normalized))
        conv2_out = self.leaky_relu2(self.conv2d_sn2(conv1_out))
        conv3_out = self.leaky_relu3(self.conv2d_sn3(conv2_out))
        conv4_out = self.leaky_relu4(self.conv2d_sn4(conv3_out))
        global_pool_out = self.flatten(self.global_avg_pool(conv4_out))
        linear1_out = self.leaky_relu5(self.linear1(global_pool_out))
        linear2_out = self.leaky_relu6(self.linear2(linear1_out))
        out = self.linear3(linear2_out)
        out = self.sigmoid(out)
        return out

class GeneratorDataset(Dataset):
    """
    A simple wrapper to feed data to the generator.
    """
    def __init__(self, file_pairs):
        super().__init__()
        self.file_pairs = file_pairs
  
    def __len__(self):
        return len(self.file_pairs)
  
    def __getitem__(self, idx):
        return get_generator_sample(self.file_pairs[idx])

class DiscriminatorDataset(Dataset):
    """
    A simple wrapper to feed data to the discriminator.
    """
    def __init__(self, file_pairs):
        super().__init__()
        self.file_pairs = file_pairs
  
    def __len__(self):
        return len(self.file_pairs)
  
    def __getitem__(self, idx):
        return get_discriminator_sample(self.file_pairs[idx])

def format_time(elapsed):
    """
    Format time deltas for printing them nicely.
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_max_checkpt(checkpt_dir=CHECKPT_DIR):
    """
    Reads in the maximum number of a checpoint in the checkpoint
    directory.
    """
    max_checkpt = 0
    for filename in os.listdir(checkpt_dir):
        if re.match(r"checkpt-gen-([0-9]+).pt", filename):
            checkpt_num = int(filename.split('.')[-2].split('-')[-1])
            if checkpt_num > max_checkpt:
                max_checkpt = checkpt_num
    return max_checkpt

def load_latest_checkpt(generator, discriminator, generator_optimizer,
        discriminator_optimizer, checkpt_dir=CHECKPT_DIR):
    """
    This function loads all previously saved state into the models and
    optimizers. However, the discriminator dataloader is newly 
    created and must be explicitly returned.
    If RESUME_FROM (1-indexed) is not -1, then we resume from
    RESUME_FROM even if a higher numbered checkpoint exists.
    
    Returns the number of the checkpoint used and the 
    new discriminator dataloader.
    """
    if RESUME_FROM == -1:
        mx_checkpt = get_max_checkpt(checkpt_dir)
    else:
        mx_checkpt = RESUME_FROM
    if mx_checkpt > 0:
        gen_checkpt_file = os.path.join(checkpt_dir, \
            "checkpt-gen-{}.pt".format(mx_checkpt))
        dis_checkpt_file = os.path.join(checkpt_dir, \
            "checkpt-dis-{}.pt".format(mx_checkpt))
        genopt_checkpt_file = os.path.join(checkpt_dir, \
            "checkpt-genopt-{}.pt".format(mx_checkpt))
        disopt_checkpt_file = os.path.join(checkpt_dir, \
            "checkpt-disopt-{}.pt".format(mx_checkpt))
        
        generator.load_state_dict(torch.load(gen_checkpt_file, \
            map_location=device))
        discriminator.load_state_dict(torch.load(dis_checkpt_file, \
            map_location=device))
        generator_optimizer.load_state_dict(torch.load(genopt_checkpt_file, \
            map_location=device))
        discriminator_optimizer.load_state_dict(torch.load(disopt_checkpt_file, \
            map_location=device))
        
        new_pairs = pickle.load(open(os.path.join(CHECKPT_DIR, \
            "npairs_{}.pkl".format(mx_checkpt)), 'rb'))
        if REPLACE:
            n = []
            dir0 = os.path.join(DATASET_DIR, "train/clean/")
            dir1 = os.path.join(CHECKPT_DIR, "epoch{}/".format(mx_checkpt-1))
            dir1_high = os.path.join(DATASET_DIR, "train/noisy/")
            for pair in new_pairs:
                name0 = os.path.join(dir0, os.path.basename(pair[0]))
                if "train/noisy" in pair[1]:
                    name1 = os.path.join(dir1_high, os.path.basename(pair[1]))
                else:
                    name1 = os.path.join(dir1, os.path.basename(pair[1]))
                n.append((name0, name1))
            new_pairs = n
        discriminator_dataset = DiscriminatorDataset(new_pairs)
        discriminator_sampler = RandomSampler(discriminator_dataset)
        discriminator_dataloader = DataLoader(discriminator_dataset, \
            sampler=discriminator_sampler, batch_size=BATCH_SIZE)
    return mx_checkpt, discriminator_dataloader

def load_latest_checkpt_only_generator(generator, checkpt_dir=CHECKPT_DIR):
    """
    Same as load_latest checkpt, but loads only the generator state.
    """
    if RESUME_FROM == -1:
        mx_checkpt = get_max_checkpt(checkpt_dir)
    else:
        mx_checkpt = RESUME_FROM
    if mx_checkpt > 0:
        gen_checkpt_file = os.path.join(checkpt_dir, \
            "checkpt-gen-{}.pt".format(mx_checkpt))        
        generator.load_state_dict(torch.load(gen_checkpt_file, \
            map_location=device))

def get_models_and_maybe_optimizers(get_opts=True, log=False):
    """
    What the name says.
    """
    generator = Generator()
    discriminator = Discriminator()
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
    if get_opts:
        generator_optimizer = AdamW( \
            generator.parameters(), lr=1e-4, eps=1e-11)
        discriminator_optimizer = AdamW( \
            discriminator.parameters(), lr=2e-5, eps=1e-11)
        return generator, discriminator, generator_optimizer, \
            discriminator_optimizer
    else:
        return generator, discriminator
        
def train(generator, discriminator, generator_optimizer, 
          discriminator_optimizer, file_pairs, clean_pairs, 
          high_pairs, log=False):
    """
    Perform the actual training of the generator and discriminator.
    If log is true, the following information is printed:
    For discriminator: 1 in every 10 examples, we print
    <predicted stoi> v/s <actual stoi with original file> v/s
    <stoi v/s original audio reconstructed (lossily) from batch inputs>
    For generator: 1 in every 10 examples we print:
    Discriminator-predicted STOI v/s real STOI
    """
    generator_dataset = GeneratorDataset(file_pairs)
    discriminator_dataset = DiscriminatorDataset(file_pairs + clean_pairs)
    generator_sampler = RandomSampler(generator_dataset)
    discriminator_sampler = RandomSampler(discriminator_dataset)
    generator_dataloader = DataLoader(generator_dataset, \
        sampler=generator_sampler, batch_size=BATCH_SIZE)
    discriminator_dataloader = DataLoader(discriminator_dataset, \
        sampler=discriminator_sampler, batch_size=BATCH_SIZE)
    if FORCE_RESTART:
        start_epoch = 0
    elif CONTINUE:
        # Start from RESUME_FROM without loading state
        start_epoch = RESUME_FROM
    else:
        start_epoch, discriminator_dataloader = \
            load_latest_checkpt(generator, discriminator, generator_optimizer, \
            discriminator_optimizer) # 0-indexed
    for gan_epoch in range(start_epoch, NUM_GAN_EPOCHS):
        print("<<<<<<<<<<<<<<< GAN epoch {} >>>>>>>>>>>>>>>>>"\
            .format(gan_epoch+1))
        discriminator.train()
        for epoch in range(NUM_DISCRIMINATOR_EPOCHS):
            val_sum = 0
            epoch_loss = 0
            epoch_start = time.time()
            true_stoi_avg = 0
            print("=============== Discriminator Epoch {} / {} =================".\
                format(epoch+1, NUM_DISCRIMINATOR_EPOCHS))
            for step, batch in enumerate(discriminator_dataloader):
                discriminator.zero_grad()
                input_noisy_discriminator = batch[0].to(device)
                expected_out_noisy = batch[1].to(device)

                outputs_noisy = discriminator(input_noisy_discriminator)
                MSE = nn.MSELoss(reduction='sum')
                loss = MSE(outputs_noisy, expected_out_noisy)
                epoch_loss += loss
                loss.backward()
                val_sum += outputs_noisy[0][0]
                clip_grad_norm_(discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
                true_stoi_avg += batch[1]
                if step % 10 == 0 and step != 0:
                    elapsed = format_time(time.time() - epoch_start)
                    if log:
                        noisy = input_noisy_discriminator[0,0,:,:].cpu().\
                            detach().numpy()
                        clean = input_noisy_discriminator[0,1,:,:].cpu().\
                            detach().numpy()
                        x = batch[2][0,:,:].cpu().detach().numpy()
                        y = batch[3][0].cpu().detach().numpy()
                        noisy_wav = spectrogram_to_wav(noisy.T, x, y) / \
                            SCALE_FACTOR
                        clean_wav = spectrogram_to_wav(clean.T, x, y)
                        print("Sample: {} v/s {} v/s {}".format( \
                            outputs_noisy[0][0], expected_out_noisy[0][0], \
                            stoi(x=clean_wav, y=noisy_wav, fs_sig=16000, \
                            extended=False)))
                    print("Batch {} of {}. Elapsed {}".format(step, len(discriminator_dataloader), elapsed))
            avg_train_loss = epoch_loss / (step+1)
            true_stoi_avg = true_stoi_avg / (step+1)
            val_sum = val_sum / (step+1)
            print("Average discriminator training loss for epoch {} : {}".\
                format(epoch+1, avg_train_loss))
            print("Average True STOI for generated outputs last epoch: {}".\
                format(true_stoi_avg[0][0]))
            print("Epoch took {}".format(format_time(time.time()-epoch_start)))
            print("")

        discriminator.eval()
        generator.train()
        for epoch in range(NUM_GENERATOR_EPOCHS):
            epoch_loss = 0
            print("============= Generator Epoch {} / {} =================".\
                format(epoch+1, NUM_GENERATOR_EPOCHS))
            epoch_start = time.time()
            for step, batch in enumerate(generator_dataloader):
                generator.zero_grad()
                input_generator = batch[0].to(device)
                noisy_audio = batch[1].to(device)
                clean = batch[2].to(device)
                min_mask = batch[3].to(device)
                target = torch.tensor([[2.0]]).to(device)

                output_generator = generator(input_generator)
                mask = torch.maximum(output_generator, min_mask)
                cleaned = torch.mul(mask, noisy_audio)
                stacked = torch.unsqueeze(torch.cat((cleaned, clean), \
                    axis=0), 0)
                discriminator_output = discriminator(stacked)
                MSE = nn.MSELoss(reduction='sum')
                loss = MSE(discriminator_output, target)
                epoch_loss += loss
                loss.backward()
                clip_grad_norm_(generator.parameters(), 1.0)
                generator_optimizer.step()
                if step % 10 == 0 and step != 0:
                    elapsed = format_time(time.time() - epoch_start)
                    cleaned = cleaned.squeeze().cpu().detach().numpy()
                    if log:
                        clean = clean.squeeze().cpu().detach().numpy()
                        x = batch[4][0,:,:].cpu().detach().numpy()
                        y = batch[5][0].cpu().detach().numpy()
                        cleaned_wav = spectrogram_to_wav(cleaned.T, x, y)  / \
                            SCALE_FACTOR
                        clean_wav = spectrogram_to_wav(clean.T, x, y)
                        print("Sample: {} v/s {}".format( \
                            discriminator_output[0][0], stoi(x=clean_wav, \
                            y=cleaned_wav, fs_sig=16000, extended=False)))
                    print("Batch {} of {}. Elapsed {}".format(step, \
                        len(generator_dataloader), elapsed))
            avg_train_loss = epoch_loss / len(generator_dataloader)
            print("Average generator training loss for epoch {} : {}".format( \
                epoch+1, avg_train_loss))
            print("Epoch took {}".format(format_time(time.time()-epoch_start)))
            print("")

        print("Saving new files for next epoch")
        generator.eval()
        if not os.path.exists('{}/epoch{}'.format(CHECKPT_DIR, gan_epoch)):
            os.mkdir('{}/epoch{}'.format(CHECKPT_DIR, gan_epoch))
        new_pairs = []
        avg_stoi = 0
        for file_pair in file_pairs:
            batch = get_generator_sample(file_pair)
            input_generator = batch[0].unsqueeze(0).to(device)
            noisy_audio = batch[1].unsqueeze(0).to(device)
            min_mask = batch[3].unsqueeze(0).to(device)

            new_pair_name = (file_pair[0], get_path_for_generator( \
                file_pair[1], gan_epoch))
            output_generator = generator(input_generator)
            mask = torch.maximum(output_generator, min_mask)
            cleaned = torch.mul(mask, noisy_audio).squeeze().cpu().detach().numpy()
            cleaned_wav = spectrogram_to_wav(cleaned.T, batch[4], batch[5]) / \
                SCALE_FACTOR
            orig_clean = librosa.load(file_pair[0], sr=16000)[0]
            s = stoi(x=orig_clean, y=cleaned_wav, fs_sig=16000, extended=False)
            avg_stoi += s
            sf.write(new_pair_name[1], cleaned_wav, 16000)
            new_pairs.append(new_pair_name)
        
        new_pairs += high_pairs
        avg_stoi /= len(file_pairs)
        print("Average True STOI: {}".format(avg_stoi))
        # New dataset for discriminator, for the next epoch
        discriminator_dataset = DiscriminatorDataset(new_pairs)
        discriminator_sampler = RandomSampler(discriminator_dataset)
        discriminator_dataloader = DataLoader(discriminator_dataset, \
            sampler=discriminator_sampler, batch_size=BATCH_SIZE)

        if save:
            torch.save(generator.state_dict(), os.path.join(CHECKPT_DIR, 
                "checkpt-gen-{}.pt".format(gan_epoch+1)))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPT_DIR, \
                "checkpt-dis-{}.pt".format(gan_epoch+1)))
            torch.save(generator_optimizer.state_dict(), os.path.join(CHECKPT_DIR, \
                "checkpt-genopt-{}.pt".format(gan_epoch+1)))
            torch.save(discriminator_optimizer.state_dict(), os.path.join( \
                CHECKPT_DIR, "checkpt-disopt-{}.pt".format(gan_epoch+1)))
            pickle.dump(new_pairs, open(os.path.join(CHECKPT_DIR, \
                "npairs_{}.pkl".format(gan_epoch+1)), 'wb+'))
    
if __name__ == '__main__':
    pass