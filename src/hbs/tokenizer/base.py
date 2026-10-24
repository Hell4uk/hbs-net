from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

class BaseTokenizer(ABC):
    """
    Contract for everyone tokenizer in project
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def pad_id(self) -> int: ...

    @property
    @abstractmethod
    def bos_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def save(self, path: Path) -> bool: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTokenizer": ...