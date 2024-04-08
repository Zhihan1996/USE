import argparse
import pickle
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Sequence
from utils import *
from modeling_retnet import RetNetModelWithLMHead
from transformers import GPT2LMHeadModel
import transformers
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from eval_user import model_paths



class CLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 dataset_dir: str, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 history_len: int = 1024,
                 context_len: int = 128,
                 predict_len: int = 128):

        super(CLMDataset, self).__init__()
        
        total_len = history_len + context_len + predict_len

        # load data from the disk
        with open(dataset_dir, "rb") as f:
            texts = pickle.load(f)
            texts = [" ".join(t.split()[-total_len:]) for t in texts]
            print(f"Number of samples: {len(texts)}")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=total_len,
            truncation=True,
        )        
        self.history_input_ids = output["input_ids"][:, :history_len]
        self.input_ids = output["input_ids"][:, history_len:]
        self.labels = self.input_ids.clone()
        if tokenizer.pad_token_id is not None:
            self.labels[self.labels == tokenizer.pad_token_id] = -100
        self.labels[:, :context_len] = -100

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(history_input_ids=self.history_input_ids[i], 
                    input_ids=self.input_ids[i], 
                    labels=self.labels[i])

    
def clm_collator(instances):
    input_ids, labels, history_input_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "history_input_ids"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    history_input_ids = torch.nn.utils.rnn.pad_sequence(
        history_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        history_input_ids=history_input_ids,
        history_attention_mask=history_input_ids.ne(tokenizer.pad_token_id),
    )
    
    

batch_size = 12
history_len = 128
context_len = 128
predict_len = 32

for history_len in [512]:
    for context_len in [0]:
        
        methods = ["retnet_v2_both_128_h4", "retnet_v2_both_128_h8", "retnet_v2_both_128_h12", "retnet_v2_both_128_h12_state"]
        tokenizer_path = "artifacts/tokenizers/bigbird_word"
        tokenizer = import_tokenizer(tokenizer_path, 
                                     bucket_name="umap-user-model", 
                                     max_model_input_size=history_len + context_len + predict_len,)
        dataset_dir = "/home/jupyter/research/research/experimental/bic/user_states/test/clm_1024_128_128_20230820_5k.pkl"
        dataset = CLMDataset(dataset_dir, tokenizer, history_len, context_len, predict_len)
        n_gpu = torch.cuda.device_count()
        data_loader = DataLoader(dataset,  batch_size=batch_size*n_gpu, collate_fn=clm_collator)


        with torch.no_grad():
            for method in methods:
                for use_history in [True]:
                    try:
                        model = import_trained_model(
                            method=method.split("_")[0],
                            model_path=model_paths[method],
                            bucket_name="umap-user-model",
                            fill_mask=False,
                            tokenizer=None,
                        )
                        model.eval()


                        n_gpu = torch.cuda.device_count()
                        device = "cuda" if n_gpu > 0 else "cpu"
                        if n_gpu > 0:
                            model.to(device)
                        if n_gpu > 1:
                            model = torch.nn.DataParallel(model)
                        

                        loss = 0
                        correct = 0
                        total = 0

                        for batch in tqdm(data_loader):
                            if use_history:
                                prev_states = model(input_ids=batch["history_input_ids"],
                                                    attention_mask=batch["history_attention_mask"],
                                                    forward_impl="chunkwise",).prev_states
                                output = model(input_ids=batch["input_ids"],
                                               attention_mask=batch["attention_mask"],
                                               labels=batch["labels"],
                                               prev_states=prev_states,
                                               sequence_offset=history_len,
                                               forward_impl="chunkwise",)

                            else:
                                output = model(input_ids=batch["input_ids"],
                                               attention_mask=batch["attention_mask"],
                                               labels=batch["labels"],)
                            loss += output.loss.mean().item()
                            logits = output.logits.cpu()[:, -predict_len:, :]
                            preds = torch.argmax(logits, dim=-1)
                            labels = batch["labels"].cpu()[:, -predict_len:]
                            preds, labels = preds[:, :-1], labels[:, 1:] # shift position
                            correct += torch.sum(preds == labels).item()
                            total += labels.shape[0] * labels.shape[1]


                        loss /= len(data_loader)
                        acc = correct / total

                        with open("user_clm.txt", "a") as f:
                            message = f"{method}_{history_len}_{context_len}_{predict_len}_history{use_history}: loss: {loss}, acc: {acc}"
                            print(message)
                            f.write(message + "\n")
                    except:
                        print(f"failed: {method}_{history_len}_{context_len}_{predict_len}_history{use_history}")
                        continue