import json
import os
from collections import defaultdict
from typing import Optional, List, Dict

import PIL
import nltk
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from vocabulary import Vocabulary


class DataLoaderWrapper:
    def __init__(
        self,
        transform,
        batch_size_for_training: int = 1,
        vocab_threshold: Optional[int] = None,
        vocab_from_file: bool = True,
        num_workers: int = 0,
        vocab_file: str = "./vocab.pkl",
        start_word: str = "<start>",
        end_word: str = "<end>",
        unk_word: str = "<unk>",
        cocoapi_loc: str = "/opt",
    ):
        self.num_workers: int = num_workers

        self.dataset_for_training = CoCoDataSetForTraining(
            transform=transform,
            batch_size=batch_size_for_training,
            vocab_threshold=vocab_threshold,
            vocab_from_file=vocab_from_file,
            vocab_file=vocab_file,
            start_word=start_word,
            end_word=end_word,
            unk_word=unk_word,
            annotations_file=os.path.join(
                cocoapi_loc, "cocoapi/annotations/captions_train2014.json"
            ),
            img_folder=os.path.join(cocoapi_loc, "cocoapi/images/train2014/"),
        )

        self.dataset_for_testing = CoCoDataSetForTesting(
            transform=transform,
            batch_size=1,  # This is fixed for testing
            vocab_threshold=vocab_threshold,
            vocab_file=vocab_file,
            start_word=start_word,
            end_word=end_word,
            unk_word=unk_word,
            annotations_file=os.path.join(
                cocoapi_loc, "cocoapi/annotations/image_info_test2014.json"
            ),
            vocab_from_file=vocab_from_file,
            img_folder=os.path.join(cocoapi_loc, "cocoapi/images/test2014/"),
        )

    def get_data_loader_for_training(self):
        return data.DataLoader(
            dataset=self.dataset_for_training,
            num_workers=self.num_workers,
            batch_sampler=data.sampler.BatchSampler(
                sampler=data.sampler.SubsetRandomSampler(
                    # Each data loader focuses on one random-sampled caption length
                    indices=self.dataset_for_training.get_train_indices()
                ),
                batch_size=self.dataset_for_training.batch_size,
                drop_last=False,
            ),
        )

    def get_data_loader_for_testing(self):
        return data.DataLoader(
            dataset=self.dataset_for_testing,
            batch_size=self.dataset_for_testing.batch_size,  # this is set to 1
            shuffle=True,
            num_workers=self.num_workers,
        )


class CoCoDataSetForTraining(data.Dataset):
    def __init__(
        self,
        transform,
        batch_size: int,
        vocab_threshold: int,
        vocab_from_file: bool,
        vocab_file: str,
        start_word: str,
        end_word: str,
        unk_word: str,
        annotations_file: str,
        img_folder: str,
    ):
        self.transform = transform
        self.batch_size: int = batch_size
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_from_file,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
        )
        self.img_folder: str = img_folder

        print("Preparing data")
        self.coco = COCO(annotations_file)
        self.annotation_indices: List[int] = list(self.coco.anns.keys())[:1000]
        self.image_by_annotation_index: Dict[int, PIL.Image] = {}
        self.caption_lengths: List[int] = []
        self.caption_tokens_by_annotation_index: Dict[int, torch.LongTensor] = {}
        self.annotation_indices_by_caption_length: Dict[int, List[int]] = defaultdict(
            list
        )

        for idx in tqdm(np.arange(len(self.annotation_indices))):
            annotation_index = self.annotation_indices[idx]

            # Caption handling: load caption, apply nltk, and then convert them to tokens
            caption = self.coco.anns[annotation_index]["caption"]
            caption_words = nltk.tokenize.word_tokenize(str(caption).lower())
            caption_tokens = [self.vocab(self.vocab.start_word)]
            caption_tokens.extend(
                [self.vocab(caption_word) for caption_word in caption_words]
            )
            caption_tokens.append(self.vocab(self.vocab.end_word))
            self.caption_lengths.append(len(caption_tokens))
            self.annotation_indices_by_caption_length[len(caption_tokens)].append(
                annotation_index
            )
            self.caption_tokens_by_annotation_index[annotation_index] = torch.Tensor(
                caption_tokens
            ).type(torch.LongTensor)

            # Image handling: load image and convert image to tensor and pre-process using transform
            image_id = self.coco.anns[self.annotation_indices[idx]]["image_id"]
            image_path = self.coco.loadImgs(image_id)[0]["file_name"]
            image = Image.open(os.path.join(self.img_folder, image_path)).convert("RGB")
            image = self.transform(image)
            self.image_by_annotation_index[annotation_index] = image

    def get_train_indices(self):
        sampled_caption_length = np.random.choice(self.caption_lengths)
        print(f"\nSampled caption length: {sampled_caption_length}", flush=True)
        return list(
            np.random.choice(
                self.annotation_indices_by_caption_length[sampled_caption_length],
                size=self.batch_size,
            )
        )

    def __getitem__(self, annotation_index):
        # obtain image and caption if in training mode
        return (
            self.image_by_annotation_index[annotation_index],
            self.caption_tokens_by_annotation_index[annotation_index],
        )

    def __len__(self):
        return len(self.annotation_indices)


class CoCoDataSetForTesting(data.Dataset):
    def __init__(
        self,
        transform,
        batch_size: int,
        vocab_threshold: int,
        vocab_from_file: bool,
        vocab_file: str,
        start_word: str,
        end_word: str,
        unk_word: str,
        annotations_file: str,
        img_folder: str,
    ):
        self.transform = transform
        self.batch_size = batch_size
        # TODO: remove vocab from testing, since it is not used
        self.vocab = Vocabulary(
            vocab_threshold,
            vocab_from_file,
            vocab_file,
            start_word,
            end_word,
            unk_word,
            annotations_file,
        )
        self.img_folder = img_folder

        # TODO: optimize this as the training one
        test_info = json.loads(open(annotations_file).read())
        self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        path = self.paths[index]

        # Convert image to tensor and pre-process using transform
        pil_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
        original_image = np.array(pil_image)
        image = self.transform(pil_image)

        # return original image and pre-processed image tensor
        return original_image, image

    def __len__(self):
        return len(self.paths)
