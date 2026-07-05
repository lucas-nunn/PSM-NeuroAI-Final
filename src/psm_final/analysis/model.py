import re

import numpy as np

from PIL import Image
from pathlib import Path

from psm_final.analysis.correlating import correlation_rdm


class ModelAnalysisBase():
    def __init__(self, triple_n_path):
        self.triple_n_path = triple_n_path

    def embedding(self, image):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def rdm(self, indices=None):
        self.shared_stimuli_dir = Path(f'{self.triple_n_path}/others/StimuliNNN')
        digit_only = re.compile(r"^\d+$")

        self.images = [
            p for p in self.shared_stimuli_dir.glob("*.bmp")
            if digit_only.fullmatch(p.stem)
        ]

        if indices is not None:
            self.images = [self.images[i] for i in indices]

        embeddings = np.stack([self.embedding(Image.open(img_path)) for img_path in self.images], axis=0)
        return correlation_rdm(embeddings)

    def rsa_algonauts(self):
        pass

    def rsa_triple_n(self):
        pass