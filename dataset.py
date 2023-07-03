import os
import re
from io import StringIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

from utils import binarise_labels, preprocessing_for_bert


class ImageCaptionDataset(Dataset):
    """
    Reference: https://github.com/alexmirrington/coco-multilabel-classification
    Dataset class for loading image-caption pairs.
    """

    TIERS = ('train', 'test')
    CLASSES = tuple(range(20))

    def __init__(
            self,
            path,
            tier,
            embeddings=None,
            preprocessor=None,
            transform=None,
            tta=False):
        super().__init__()
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, f'{tier}.csv'))
        assert tier in self.TIERS

        self.path = path
        self.tier = tier
        self.data = None
        self.embeddings = embeddings
        self.transform = transform
        self.tta = tta
        self.tta_time = 10
        self._init_dataset()

    def _init_dataset(self):
        # Read data and preprocess
        with open(os.path.join(self.path, f'{self.tier}.csv')) as file:
            lines = [re.sub(
                r'([^,])"(\s*[^\n])',
                r'\1`"\2',
                line
            ) for line in file]
            self.data = pd.read_csv(
                StringIO(''.join(lines)),
                escapechar='`'
            )

        # Preprocess labels
        if self.tier == self.TIERS[0]:
            lbl_col = self.data.columns[1]
            self.data[lbl_col] = self.data[lbl_col].apply(
                lambda lbls: [int(lbl) for lbl in lbls.split()]
            )
            binarised, _ = binarise_labels(
                list(self.data[lbl_col]),
                classes=self.CLASSES
            )
            binarised = [torch.Tensor(vec) for vec in binarised]
            self.data[lbl_col] = binarised
            self.text_label = np.stack(binarised)

        # Preprocess captions and tokenize
        caption_col = self.data.Caption.values
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        print('Tokenizing data...')
        self.text_input, self.text_masks = preprocessing_for_bert(caption_col, tokenizer)
        print("Text processing done")

    def __getitem__(self, key):
        """
        get item method for the dataset. returns a tuple of the image id, image and the caption by the key.
        """
        # Get images and apply transforms
        image_file = self.data[self.data.columns[0]][key]
        image = Image.open(os.path.join(self.path, 'data', image_file))
        if self.transform:
            original_img = image

            # Test time augmentation
            if self.tta:
                image = []
                for i in range(self.tta_time):
                    image.append(self.transform(original_img))
            else:
                image = self.transform(image)

        # When it is training set
        if self.tier == self.TIERS[0]:
            labels = self.data[self.data.columns[1]][key]
            return image_file, image, tuple([self.text_input[key], self.text_masks[key]]), labels

        # when it is val or test set
        return image_file, image, tuple([self.text_input[key], self.text_masks[key]])

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data)
