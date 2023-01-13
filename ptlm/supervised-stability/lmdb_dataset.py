from typing import Iterable, Tuple, Any, Set, TypeVar, Callable, List
from functools import partial
from pathlib import Path
import pickle as pkl
import gzip
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
from evo.typed import PathLike
from evo.dataset import ThreadsafeFile


RESERVED_KEY = "__lmdb_keys"
COMPRESS_KEY = "__lmdb_compress"

T = TypeVar("T")


def write_lmdb(
    filename: str,
    iterable: Iterable[Tuple[str, T]],
    map_size: int = 2 ** 20,
    save_func: Callable[[T], bytes] = pkl.dumps,
    compress: bool = False,
):
    """Utility for writing a dataset to an LMDB file.

    Args:
        filename (str): Output filename to write to
        iterable (Iterable): An iterable dataset to write to. Entries must be
            pickleable.
        map_size (int, optional): Maximum allowable size of database in bytes. Required
            by LMDB. You will likely have to increase this. Default: 1MB.
    """
    env = lmdb.open(filename, map_size=map_size)

    def save(x: T) -> bytes:
        data = save_func(x)
        if compress:
            data = gzip.compress(data)
        return data

    keys: Set[str] = set()
    with env.begin(write=True) as txn:
        for key, value in tqdm(iterable):
            if key in keys or key == RESERVED_KEY:
                raise ValueError(f"Duplicate key in dataset: {key}")
            txn.put(key.encode(), save(value))
            keys.add(key)
        data = pkl.dumps(keys)
        if compress:
            data = gzip.compress(data)
        txn.put(RESERVED_KEY.encode(), data)
        txn.put(COMPRESS_KEY.encode(), str(int(compress)).encode())
    env.close()


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(
        self, data_file: PathLike, load_func: Callable[[bytes], T] = pkl.loads
    ):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        self.env = ThreadsafeFile(
            str(data_file),
            partial(lmdb.open, readonly=True, lock=False),
        )
        self.load_func = load_func

        with self.env.begin(write=False) as txn:
            self.compressed = bool(int(txn.get(COMPRESS_KEY.encode()).decode()))
            self.decompress = gzip.decompress if self.compressed else lambda x: x
            self.keys: List[str] = sorted(list(
                pkl.loads(self.decompress(txn.get(RESERVED_KEY.encode())))
            ))

    def __len__(self) -> int:
        return len(self.keys)

    def get(self, key: str) -> T:
        with self.env.begin(write=False) as txn:
            data = txn.get(key.encode())
            item = self.load_func(self.decompress(data))
        return item

    def __getitem__(self, index: int) -> T:
        if not (0 <= index < len(self)):
            raise IndexError(index)

        key = self.keys[index]
        with self.env.begin(write=False) as txn:
            data = txn.get(key.encode())
            item = self.load_func(self.decompress(data))
        return item
