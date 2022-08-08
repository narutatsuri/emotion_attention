from util import *
from util.functions import *
from util.models import *


# Load ISEAR dataset
dataset = process_isear_dataset()

# Load language model
model = language_model(pretrained_model)
# Get embeddings for all layers for all sentences
embedding = model.embed_sentences(list(dataset.keys()))