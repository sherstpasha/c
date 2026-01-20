from CharLM.config import DEFAULT_CONFIG
from CharLM.train import train

CONFIG = {**DEFAULT_CONFIG}

if __name__ == "__main__":
    train(CONFIG)
