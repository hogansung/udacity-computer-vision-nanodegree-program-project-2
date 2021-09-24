import os.path
import pickle
from collections import Counter
from typing import Optional, Dict

import nltk
from pycocotools.coco import COCO


class Vocabulary(object):
    def __init__(
        self,
        vocab_threshold: Optional[int] = None,
        vocab_from_file: bool = False,
        vocab_file: str = "./vocab.pkl",
        start_word: str = "<start>",
        end_word: str = "<end>",
        unk_word: str = "<unk>",
        annotations_file: str = "../cocoapi/annotations/captions_train2014.json",
    ):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold: Optional[int] = vocab_threshold
        self.vocab_from_file: bool = vocab_from_file
        self.vocab_file: str = vocab_file
        self.start_word: str = start_word
        self.end_word: str = end_word
        self.unk_word: str = unk_word
        self.annotations_file: str = annotations_file
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_idx: int = 0
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def add_word(self, word: str):
        """Add a token to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.word_idx
            self.idx2word[self.word_idx] = word
            self.word_idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        annotation_indices = coco.anns.keys()
        for idx, annotation_index in enumerate(annotation_indices):
            caption = str(coco.anns[annotation_index]["caption"])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if idx % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (idx, len(annotation_indices)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word: str):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
