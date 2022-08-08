import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class language_model():
    """
    Class for deep language model.
    Embedding code by Kohki Horie (khorie4900@g.ecc.u-tokyo.ac.jp).
    """
    def __init__(self, 
                 checkpoint):
        """
        Constructor for language_model.
        INPUTS:
            checkpoint (string): Pretrained checkpoint.
                                 Search at https://huggingface.co/models.
        """
        # Set device for loading model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, 
                                               output_hidden_states=True).to(device)
        
    def get_embeddings(self, 
                       sequence):
        """
        INPUTS:     
            
        RETURNS:
            
        """
        model_inputs = self.tokenizer(sequence, 
                                      truncation=True, 
                                      return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**model_inputs)
        
        return output.hidden_states
    
    def embed_sentences(sentences):    
        """
        Returns embeddings of <CLS> token for each layer. 
        INPUTS:     
            sentences (list[string]): List of sentences to pass through model.
        RETURNS:
            out (numpy 3d ndarray): All embeddings of size 
                                    len(texts) * 
                                    (number of encoders + 1 = number of layers) * 
                                    (length of embedded vector).
        """
        n_sentences = len(sentence)

        embeddings = [], [[] for _ in range(n_layers + 1)]

        for i in tqdm.tqdm(range(n_sentences)):
            sentence = sentences[i]
            embeddings_tmp = self.get_embeddings(sentence)
            
            for i in range(n_layers + 1):
                embeddings_layer_tmp = embeddings_tmp[i].cpu().tolist()[0]
                embeddings_layer_tmp = np.array(embeddings_layer_tmp)
                embeddings[i].extend(embeddings_layer_tmp[[0]]) 

        embeddings = np.array(embeddings)
        
        return embeddings
