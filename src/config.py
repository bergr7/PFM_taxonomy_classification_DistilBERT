"""
Config file.

"""
from torch import cuda
from transformers import DistilBertTokenizer

INFERENCE_FILE = '../data/test_inference.csv'
TAXONOMY_MODEL_PATH = '../input/taxonomy_approach_1_v5.bin'
SUBTAXONOMY_MODEL_PATH = '../input/subtaxonomy_approach_1_v1.bin'

TAXONOMY_MAX_LEN = 210
SUBTAXONOMY_MAX_LEN = 150
BATCH_SIZE = 16
SEED = 42

TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# dataloader params
PARAMS = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'num_workers': 4
          }

# Setting up the device for GPU usage if available
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
print(DEVICE)
