import os, glob
import torch.utils.data as data
from transformers import Wav2Vec2FeatureExtractor
from random import randint
import librosa
import enum
import numpy as np
import pandas
import copy
import torch
import math
import soundfile

MOSQUITO_LABEL = 1
BACKGROUND_LABEL = 0
SAMPLE_RATE = 16_000
MAX_SECONDS = 40
EXTENSION_EXTERNAL_NOISE = "wav"
FRAME_DURATION = 20.0001


class MaskingType(enum.Enum):
    no_mask = 1
    mask_start = 2
    mask_central = 3
    mask_end = 4


class MosTimestampDataset(data.Dataset):
    def __init__(
        self,
        train_df,
        external_noise_folder,
        root_dataset_folder,
        model_name_or_path="microsoft/wavlm-base-plus-sd",
        num_labels=3,
        augmentation=True,
    ):
        self.files_id = list(train_df.id)
        self.labels = list(train_df.label)
        self.train_df = train_df
        self.root_dataset_folder = root_dataset_folder
        self.external_noise_folder = external_noise_folder
        self.model_name_or_path = model_name_or_path
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name_or_path
        )
        self.num_labels = num_labels

        if augmentation:
            self.prob_masking = [0.25, 0.25, 0.25, 0.25]
        else:
            self.prob_masking = [1, 0.0, 0.0, 0.0]

    def __getitem__(self, index):

        masking_type_list = [
            MaskingType.no_mask,
            MaskingType.mask_start,
            MaskingType.mask_central,
            MaskingType.mask_end,
        ]

        masking_type = np.random.choice(masking_type_list, 1, p=self.prob_masking)[0]

        if self.labels[index] == MOSQUITO_LABEL:
            filename = f"{self.root_dataset_folder}/{self.files_id[index]}.wav"
            mos_audio, _ = librosa.load(filename, sr=SAMPLE_RATE)

            if masking_type == MaskingType.no_mask:

                total_duration_ms = MAX_SECONDS * 1000
                total_bins = [0] * int(total_duration_ms / FRAME_DURATION)
                mos_duration = len(mos_audio) / (SAMPLE_RATE / 1000)

                if self.num_labels == 3:
                    mos_bins = [1] + int(
                        (mos_duration - FRAME_DURATION) / FRAME_DURATION
                    ) * [2]
                elif self.num_labels == 4:
                    mos_bins = (
                        [1]
                        + int((mos_duration - 2 * FRAME_DURATION) / FRAME_DURATION)
                        * [2]
                        + [3]
                    )
                elif self.num_labels == 2:
                    mos_bins = [1] * int((mos_duration) / FRAME_DURATION)
                else:
                    print(
                        f"Error: num_labels must be an integer value among [2, 3, 4]. Current value: {self.num_labels}, using num_labels = 3."
                    )
                    mos_bins = [1] + int(
                        (mos_duration - FRAME_DURATION) / FRAME_DURATION
                    ) * [2]

                inputs = self.feature_extractor(
                    [mos_audio],
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                    padding="max_length",
                    max_length=MAX_SECONDS * SAMPLE_RATE,
                    truncation=True,
                )

                insert_index = 0
                total_bins[insert_index : insert_index + len(mos_bins)] = mos_bins

            else:
                is_internal_noise = np.random.choice([True, False], 1, p=[0.25, 0.75])[
                    0
                ]
                if is_internal_noise:
                    try:
                        noise_id = (
                            self.train_df[self.train_df.label == BACKGROUND_LABEL]
                            .sample(1)["id"]
                            .tolist()[0]
                        )
                        noise_filename = f"{self.root_dataset_folder}/{noise_id}.wav"
                        noise_audio, _ = librosa.load(noise_filename, sr=SAMPLE_RATE)
                        noise_factor = np.random.uniform(0.2, 0.5, 1)[0]
                    except:
                        print(f"Internal noise {noise_id} not found.")
                        list_files = list(
                            glob.glob(
                                f"{self.root_dataset_folder}/*.{EXTENSION_EXTERNAL_NOISE}"
                            )
                        )
                        noise_filename = np.random.choice(list_files, 1)[0]
                        noise_audio, _ = librosa.load(noise_filename, sr=SAMPLE_RATE)
                else:
                    list_files = list(
                        glob.glob(
                            f"{self.external_noise_folder}/*.{EXTENSION_EXTERNAL_NOISE}"
                        )
                    )
                    noise_filename = np.random.choice(list_files, 1)[0]
                    noise_audio, _ = librosa.load(noise_filename, sr=SAMPLE_RATE)
                    noise_factor = np.random.uniform(0.1, 0.3, 1)[0]

                if len(noise_audio) <= len(mos_audio):
                    repeats = math.ceil(len(mos_audio) / len(noise_audio))
                    noise_signal = copy.deepcopy(noise_audio)
                    for n in range(repeats - 1):
                        noise_signal = np.concatenate((noise_signal, noise_audio))
                    noise_audio = noise_signal

                if masking_type == MaskingType.mask_start:
                    mos_start_time_ms = 0
                elif masking_type == MaskingType.mask_central:
                    range_start = list(
                        range(
                            0,
                            int(
                                len(noise_audio) / (SAMPLE_RATE / 1000)
                                - len(mos_audio) / (SAMPLE_RATE / 1000)
                            ),
                        )
                    )
                    if len(range_start) < 1:
                        mos_start_time_ms = 0
                    else:
                        mos_start_time_ms = np.random.choice(range_start, 1)[0]
                elif masking_type == MaskingType.mask_end:
                    mos_start_time_ms = int(
                        len(noise_audio) / (SAMPLE_RATE / 1000)
                        - len(mos_audio) / (SAMPLE_RATE / 1000)
                    )
                else:
                    print("Error: unrecognized masking type, using MaskingType.central")
                    range_start = list(
                        range(
                            0,
                            int(
                                len(noise_audio) / (SAMPLE_RATE / 1000)
                                - len(mos_audio) / (SAMPLE_RATE / 1000)
                            ),
                        )
                    )
                    mos_start_time_ms = np.random.choice(range_start, 1)[0]

                zeros_pre = int(mos_start_time_ms * (SAMPLE_RATE / 1000))
                end_time = mos_start_time_ms + int(
                    len(mos_audio) / (SAMPLE_RATE / 1000)
                )
                zeros_post = int(
                    (len(noise_audio) / (SAMPLE_RATE / 1000) - end_time)
                    * (SAMPLE_RATE / 1000)
                )

                zeros_pre = np.zeros(zeros_pre)
                zeros_post = np.zeros(zeros_post)
                mos_audio_padded = np.concatenate((zeros_pre, mos_audio, zeros_post))

                try:
                    assert mos_audio_padded.shape == noise_audio.shape
                except Exception as e:
                    # print (f"Error different shapes MOS:{ mos_audio_padded.shape}, BACKGROUND:{noise_audio.shape}")
                    # print (f"Trimming the longer")
                    if len(mos_audio_padded) > len(noise_audio):
                        mos_audio_padded = mos_audio_padded[: len(noise_audio)]
                    else:
                        noise_audio = noise_audio[: len(mos_audio_padded)]

                augmented_signal = mos_audio_padded + noise_audio * noise_factor

                #soundfile.write(f"temp_augmentation/{self.files_id[index]}_signal.wav", mos_audio, SAMPLE_RATE)
                #soundfile.write(f"temp_augmentation/{self.files_id[index]}_noise.wav", noise_audio, SAMPLE_RATE)
                #soundfile.write(f"temp_augmentation/{self.files_id[index]}_augmented.wav", augmented_signal, SAMPLE_RATE)

                # pass to the Feature Extractor
                inputs = self.feature_extractor(
                    [augmented_signal],
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                    padding="max_length",
                    max_length=MAX_SECONDS * SAMPLE_RATE,
                    truncation=True,
                )

                # compute labels
                total_duration_ms = MAX_SECONDS * 1000
                total_bins = [0] * int(total_duration_ms / FRAME_DURATION)

                mos_duration = len(mos_audio) / (SAMPLE_RATE / 1000)
                if self.num_labels == 3:
                    mos_bins = [1] + int(
                        (mos_duration - FRAME_DURATION) / FRAME_DURATION
                    ) * [2]
                elif self.num_labels == 4:
                    mos_bins = (
                        [1]
                        + int((mos_duration - 2 * FRAME_DURATION) / FRAME_DURATION)
                        * [2]
                        + [3]
                    )
                elif self.num_labels == 2:
                    mos_bins = [1] * int((mos_duration) / FRAME_DURATION)
                else:
                    print(
                        f"Error: num_labels must be an integer value among [2, 3, 4]. Current value: {self.num_labels}, using num_labels = 3."
                    )

                insert_index = int(mos_start_time_ms / FRAME_DURATION)
                total_bins[insert_index : insert_index + len(mos_bins)] = mos_bins
        else:
            # Background
            filename = f"{self.root_dataset_folder}/{self.files_id[index]}.wav"
            background_audio, _ = librosa.load(filename, sr=SAMPLE_RATE)
            total_duration_ms = MAX_SECONDS * 1000
            total_bins = [0] * int(total_duration_ms / FRAME_DURATION)
            inputs = self.feature_extractor(
                [background_audio],
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
                padding="max_length",
                max_length=MAX_SECONDS * SAMPLE_RATE,
                truncation=True,
            )

        total_duration_ms = MAX_SECONDS * 1000
        max_bins_len = int(total_duration_ms / FRAME_DURATION)
        if len(total_bins) > max_bins_len:
            total_bins = total_bins[:max_bins_len]
        labels = []
        for v in total_bins:
            temp_list = [0] * self.num_labels
            temp_list[v] = 1
            labels.append(temp_list)

        item = {}
        item["input_values"] = inputs["input_values"][0]
        item["attention_mask"] = inputs["attention_mask"][0]
        item["labels"] = torch.as_tensor(labels)
        return item

    def __len__(self):
        return len(self.labels)
