import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def calc_embedding(checkpoint, texts):    
    """
    returns embeddings of <CLS> token for each layer
    
    Parameters
    ----------
    checkpoint : checkpoint
                 search at https://huggingface.co/models
    texts : texts
    
    Returns
    -------
    out : 3d ndarray
          size len(texts) * (number of encoders + 1 = number of layers) * (length of embedded vector)
    
    
    Examples
    --------
    >>> calc_embedding("bert-base-uncased", ["hello world", "Hello World!", "foo bar"])
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, output_hidden_states=True)
    model = model.to(device)
 
    def get_embeddings(sequence):
        model_inputs = tokenizer(sequence, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**model_inputs)
        words = [tokenizer.decode(_) for _ in tokenizer.encode(sequence)]
        return words, output.hidden_states


    N_layer = len(get_embeddings("hello world")[1]) - 1
    N_text = len(texts)
    print(N_layer)

    words, embeddings = [], [[] for _ in range(N_layer + 1)]

    for i in tqdm.tqdm(range(N_text)):
        text = " ".join(texts[i])
        #text = texts[i]
        print(text)
        words_tmp, embeddings_tmp = get_embeddings(text)
        words.extend(words_tmp)
        print(words)
        for i in range(N_layer + 1):
            embeddings_layer_tmp = embeddings_tmp[i].cpu().tolist()[0]
            embeddings_layer_tmp = np.array(embeddings_layer_tmp)
            print(embeddings_layer_tmp[[0]].shape)
            embeddings[i].extend(embeddings_layer_tmp[[0]]) 
            # embeddings[i].extend(embeddings_layer_tmp) uncomment this line to get embeddings of all tokens instead of only <CLS>

    embeddings = np.array(embeddings)
    
    return embeddings

if __name__ == "__main__":
    embedding = calc_embedding("bert-base-uncased", ["hello world"])
    print(embedding.shape)