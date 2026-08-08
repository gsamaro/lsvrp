import json
from pathlib import Path


class Config:
    _data = None

    @classmethod
    def load(cls):
        if cls._data is None:
            config_path = Path(__file__).parent / "config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                cls._data = json.load(f)

    @classmethod
    def get(cls, key, default=None):
        cls.load()
        return cls._data.get(key, default)

    @classmethod
    def get_nested(cls, *keys, default=None):
        cls.load()
        data = cls._data
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key)
        return data if data is not None else default
