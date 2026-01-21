from charlm.config import DEFAULT_CONFIG
from charlm.train import train

CONFIG = {**DEFAULT_CONFIG}

if __name__ == "__main__":
    train(CONFIG)
