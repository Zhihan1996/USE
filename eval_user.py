import argparse
import pickle
import torch.nn.functional as F
import numpy as np
from utils import *

pooling_modes = {
    "tf": ["na"],
    "tf-l2": ["na"],
    "tf-idf": ["na"],
    "n-grams": ["na"],
    "sgns": ["mean"],
    "random": ["mean"],
    "bert": ["mean"],
    "gpt2": ["mean"],
    "retnet": ["mean"],
    "rwkv": ["mean"],
}

model_paths = {
    "sgns": "artifacts/sgns",
    "bert_CHI_SUP_AT": "artifacts/CHI/SUP_AT",
    "bert_CHI_Both_AT": "artifacts/CHI/Both_AT",
    "bert_CHI_Both_AF": "artifacts/CHI/Both_AF",
    "bert_CHI_MLM_AT": "artifacts/CHI/MLM_AT",
    "bert_CHI_MLM_AF": "artifacts/CHI/MLM_AF",
    "gpt2_CHI": "artifacts/CHI/gpt2",
    "bert_aFtF_both": "artifacts/20230804/bert_aFtF_both_128",
    "bert_aTtF": "artifacts/20230804/bert_aTtF_128",
    "bert_aTtF_both": "artifacts/20230804/bert_aTtF_both_128",
    "gpt2_128": "artifacts/20230804/gpt2_128",
    "gpt2_512": "artifacts/20230804/gpt2_512",
    "retnet_v2_both_128_h3_Orig_bs8_token": "artifacts/Both_retnet_full_128_v2_h3_ep5_token_Orig_bs8retnet/kt3cvgzg/checkpoint-7000",
    "retnet_v2_both_128_h3_Orig_bs32_token": "artifacts/Both_retnet_full_128_v2_h3_ep5_token_Orig_bs32retnet/5rtmx2w5/checkpoint-7000",
    "retnet_v2_both_128_h3_simsiam": "artifacts/Both_retnet_full_128_v2_h3_ep5_token_simsiam_bs32retnet/3wdqod4f/checkpoint-7000",
    "retnet_v2_both_128_h3_simsiam_noproj": "artifacts/Both_retnet_full_128_v2_h3_ep5_token_simsiam_bs32_noprojretnet/2ah03a3p/checkpoint-7000",
    "retnet_v2_128_h3": "artifacts/20230820/retnet_v2_128_h3",
    "retnet_v2_128_h4": "artifacts/20230820/retnet_v2_128_h4",
    "retnet_v2_128_h8": "artifacts/20230820/retnet_v2_128_h8",
    "retnet_v2_128_h12": "artifacts/20230820/retnet_v2_128_h12",
    "retnet_v2_both_128_h4": "artifacts/20230820/retnet_v2_both_128_h4",
    "retnet_v2_both_128_h8": "artifacts/20230820/retnet_v2_both_128_h8",
    "retnet_v2_both_128_h12": "artifacts/20230820/retnet_v2_both_128_h12",
    "retnet_v2_both_128_h12_state": "artifacts/20230820/retnet_v2_both_128_h12_state",
    "retnet_128": "artifacts/20230804/retnet_128",
    "retnet_equal_128": "artifacts/20230804/retnet_equal_128",
    "retnet_equal_both_128": "artifacts/20230804/retnet_equal_both_128",
    "retnet_equal_512": "artifacts/20230804/retnet_equal_512",
    "retnet_equal_both_512": "artifacts/20230804/retnet_equal_both_512",
    "retnet_v2_fep_h8_29": "artifacts/20230917/FEP_29_context0_split",
    "retnet_v2_fep_h8_686": "artifacts/20230917/FEP_686_context0_split",
    "bert_final": "artifacts/final/bert",
    "bert_aT_final": "artifacts/final/bert_aT",
    "gpt2_final": "artifacts/final/gpt2",
    "retnet_clm_final": "artifacts/final/retnet_CLM",
    "retnet_fep_final": "artifacts/final/retnet_FEP",
    "retnet_sup_final": "artifacts/final/retnet_SUP",
    "retnet_fepsup_final": "artifacts/final/retnet_FEP_SUP",
}



if __name__ == "__main__":
    methods = ["retnet_fep_final", "retnet_fepsup_final"]
    tokenizer_path = "artifacts/tokenizers/bigbird_word"
    dataset_dir = "/home/jupyter/research/research/experimental/bic/user_states/test/final_user_retrieval_4608_20230921_neg080_pos095_3k.pkl"

    bucket_name = "umap-user-model"
    ori_batch_size = 32
    max_model_input_size = 512
    recurrent_input_size = 512
    split_mode = "split"

    with open(dataset_dir, "rb") as f:
        raw_data = pickle.load(f)
        print(f"Number of samples: {len(raw_data)}")

    event_sequences = []
    for d in raw_data:
        event_sequences += d
    assert len(event_sequences) % 101 == 0
    
    for max_model_input_size in [4096]:     
        for last_segment_only in [False]:
            print(f"max_len {max_model_input_size}, rnn_len {recurrent_input_size}, bs {ori_batch_size}, mode {split_mode}, methods {methods}, last_segment_only{last_segment_only}")




            for method in methods:
                modes = pooling_modes[method.split("_")[0]]
                # batch_size = ori_batch_size if method in ["retnet", "gpt2"] else ori_batch_size // 2
                if max_model_input_size > 512:
                    times_l = max_model_input_size // 512
                    batch_size = ori_batch_size // times_l
                all_embeddings, _, _ = feature_engineering(train_event_sequences = event_sequences,
                                                            test_event_sequences = [],
                                                            train_time_sequences = [],
                                                             test_time_sequences = [],
                                                             method = method.split("_")[0],
                                                             modes = modes,
                                                             max_model_input_size = max_model_input_size,
                                                             model_path = model_paths[method] if method in model_paths else "",
                                                             tokenizer_path = tokenizer_path,
                                                             bucket_name = bucket_name,
                                                             batch_size = batch_size,
                                                             recurrent_input_size = recurrent_input_size,
                                                             split_mode = split_mode,
                                                             reverse_sequence = True,
                                                             last_segment_only = last_segment_only,
                                                            )

                def get_rank(x):
                    vals = x[range(len(x)), 0]
                    return (x > vals[:, None]).long().sum(1) + 1


                for mode in modes:
                    embeddings = all_embeddings[mode]
                    embeddings = torch.tensor(embeddings) if type(embeddings) == list else torch.tensor(np.array(embeddings))
                    embeddings = F.normalize(embeddings.float())
                    feature_dim = embeddings.shape[-1]
                    embeddings = embeddings.reshape([-1, 101, feature_dim])
                    embeddings_query = embeddings[:, 0, :].unsqueeze(1) # [batch_size, 1, hidden_size] 
                    embeddings_candidate = embeddings[:, 1:, :] # [batch_size, 100, hidden_size] 
                    similarities = torch.einsum('bih,bjh->bij', embeddings_query, embeddings_candidate).squeeze(1) # [batch_size, 100] 
                    ranks = get_rank(similarities)

                    mrr = torch.mean(1.0 / ranks)
                    top_1_acc = ((ranks <= 1).sum() / len(ranks)).item()
                    top_3_acc = ((ranks <= 3).sum() / len(ranks)).item()
                    top_5_acc = ((ranks <= 5).sum() / len(ranks)).item()
                    top_10_acc = ((ranks <= 10).sum() / len(ranks)).item()

                    with open("user_retriveal.txt", "a") as f:
                        message = f"{method}_{mode}_{max_model_input_size}: mrr {mrr}, top1 {top_1_acc}, top3 {top_3_acc}, top5 {top_5_acc}, top10 {top_10_acc}"
                        print(message)
                        f.write(message + "\n")
