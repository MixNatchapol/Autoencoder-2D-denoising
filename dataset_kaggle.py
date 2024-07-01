import numpy as np
import librosa
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import soundfile as sf

class SnoringDataset(Dataset):
    def __init__(self, dataset_path, split='train', val_split=0.2, test_split=0.1, snr=10, target_length=64):
        self.dataset_path = dataset_path
        self.split = split
        self.val_split = val_split
        self.test_split = test_split
        self.snr = snr
        self.target_length = target_length

        self.snoring_files = self.load_files(os.path.join(dataset_path, '1'))
        self.non_snoring_files = self.load_files(os.path.join(dataset_path, '0'))
        self.data = self.prepare_dataset()

    def load_files(self, folder_path):
        return glob.glob(os.path.join(folder_path, '*.wav'))

    def prepare_dataset(self):
        combined_clean_pairs = self.generate_combined_data(self.snoring_files, self.non_snoring_files)
        
        train_val_data, test_data = train_test_split(combined_clean_pairs, test_size=self.test_split, random_state=0)
        train_data, val_data = train_test_split(train_val_data, test_size=self.val_split / (1 - self.test_split), random_state=0)

        if self.split == 'train':
            return train_data
        elif self.split == 'val':
            return val_data
        elif self.split == 'test':
            return test_data

    def generate_combined_data(self, snore_files, noise_files):
        combined_clean_pairs = []
        save_audio_dir = os.path.join(self.dataset_path, 'combined_audio')
        save_mel_dir = os.path.join(self.dataset_path, 'combined_mel_spectrograms')
        os.makedirs(save_audio_dir, exist_ok=True)
        os.makedirs(save_mel_dir, exist_ok=True)
        
        for idx, snore_file in enumerate(tqdm(snore_files, desc="Generating combined data")):
            snore, sr_snore = self.load_wav_16k_mono(snore_file)
            noise_file = np.random.choice(noise_files)
            noise, sr_noise = self.load_wav_16k_mono(noise_file)

            if len(noise) < len(snore):
                noise = np.tile(noise, int(np.ceil(len(snore) / len(noise))))
            noise = noise[:len(snore)]

            combined_snore = self.combine_snore_noise(snore, noise, self.snr)
            combined_clean_pairs.append((combined_snore, snore, idx))
            
            combined_filename = os.path.join(save_audio_dir, f'combined_{idx}.wav')
            sf.write(combined_filename, combined_snore, 16000)
            
            snore_filename = os.path.join(save_audio_dir, f'snore_{idx}.wav')
            sf.write(snore_filename, snore, 16000)
            
            combined_mel = self.compute_mel_spectrogram(combined_snore)
            combined_mel_filename = os.path.join(save_mel_dir, f'combined_mel_{idx}.npy')
            np.save(combined_mel_filename, combined_mel)
            
            snore_mel = self.compute_mel_spectrogram(snore)
            snore_mel_filename = os.path.join(save_mel_dir, f'snore_mel_{idx}.npy')
            np.save(snore_mel_filename, snore_mel)
        
        return combined_clean_pairs

    def load_wav_16k_mono(self, filename):
        y, sr = librosa.load(filename, sr=16000, mono=True)
        return y, sr

    def combine_snore_noise(self, snore, noise, snr):
        combined = snore + noise
        combined = self.normalization(combined)
        return combined

    def normalization(self, arr):
        norm_arr = arr / np.max(np.abs(arr))
        return norm_arr

    def compute_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, hop_length=256)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = self.pad_or_crop(mel_spec_db, self.target_length)
        mel_spec_db = self.normalization(mel_spec_db)  # Ensure normalization
        return mel_spec_db

    def pad_or_crop(self, mel_spec, target_length):
        if (mel_spec.shape[1] < target_length):
            pad_width = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_length]
        return mel_spec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        combined_audio, snore_audio, idx = self.data[index]

        combined_mel_file = os.path.join(self.dataset_path, 'combined_mel_spectrograms', f'combined_mel_{idx}.npy')
        snore_mel_file = os.path.join(self.dataset_path, 'combined_mel_spectrograms', f'snore_mel_{idx}.npy')
        
        combined_mel = np.load(combined_mel_file)
        snore_mel = np.load(snore_mel_file)

        combined_mel = torch.tensor(combined_mel, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        snore_mel = torch.tensor(snore_mel, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return combined_mel, snore_mel, idx
