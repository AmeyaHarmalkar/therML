from typing import Tuple
import re
import numpy as np
from evo.tokenization import Vocab

deepab_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19',
    '<pad>': '20',
}


class DeepabVocab(Vocab):

    def __init__(self):
        super().__init__(
            deepab_aa_dict,
            prepend_bos=False,
            append_eos=False,
        )
        self.linkers = [
            "GGGGS" * 3,  # G4Sx3
            "GGGGS" * 2,  # G4Sx2
            "GGGGSGGGSGGGGS",  # G4SG3SG4S - likely mistranscription
            "GGGGSGGGGPGGGGS",  # G4SG4PG4S - likely mistranscription
        ]

    def split_linkers(self, seq: str) -> Tuple[str, int]:
        for linker in self.linkers:
            try:
                heavy, light = seq.split(linker)
            except ValueError:
                continue
            return heavy + light, len(heavy)
        else:
            raise RuntimeError(f"No appropriate linker found: {seq}")

    def encode(self, seq: str, validate: bool = False) -> np.ndarray:
        seq = re.sub("X", "", seq)
        seq, hlen = self.split_linkers(seq)
        tokens = super().encode(seq)
        array = np.eye(21, dtype=np.float32)[tokens]
        array[hlen - 1, 20] = 1
        return array.T
