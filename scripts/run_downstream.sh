#!/bin/bash

model_name="retnet_SUP retnet_FEP_SUP"
model_path="artifacts/final/retnet_SUP artifacts/final/retnet_FEP_SUP"
run_name=Final_v1_1024
max_len=1024
bs=16
tokenizer_path="artifacts/tokenizers/bigbird_word artifacts/tokenizers/bigbird_word"
sgns_model_path="artifacts/sgns"
output="../artifacts/final_1024"
baselines="tf tf-idf random sgns"

# Define lists of values for each parameter
features_start_dates=(
"20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602" 
"20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601" 
"20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531" 
"20230511 20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530" 
"20230510 20230511 20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529" 
"20230509 20230510 20230511 20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528" 
"20230508 20230509 20230510 20230511 20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527" 
"20230507 20230508 20230509 20230510 20230511 20230512 20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526")
features_end_dates=(
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607" 
"20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606" 
"20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605" 
"20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604" 
"20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603" 
"20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602" 
"20230513 20230514 20230515 20230516 20230517 20230518 20230519 20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601")
labels_dates=(
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608" 
"20230520 20230521 20230522 20230523 20230524 20230525 20230526 20230527 20230528 20230529 20230530 20230531 20230601 20230602 20230603 20230604 20230605 20230606 20230607 20230608")



# Loop through the value combinations
for ((i=0; i<${#features_start_dates[@]}; i++)); do
    features_start_date="${features_start_dates[i]}"
    features_end_date="${features_end_dates[i]}"
    labels_date="${labels_dates[i]}"
    for n_class in 2; 
    do
        python predict.py --task test_reported_user_prediction --sample-n 1000 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
        python predict.py --task test_locked_user_prediction --sample-n 1000 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
        python predict.py --task test_account_self_deletion_prediction --sample-n 1000 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
        python predict.py --task test_ad_click_binary_prediction --sample-n 1000 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --negative-label no_ad_click --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
        python predict.py --task test_ad_view_time_prediction --sample-n 2500 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --negative-label "<=2s" --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
        python predict.py --task test_ad_view_time_prediction --sample-n 2500 --features-start-dates "$features_start_date" --features-end-dates "$features_end_date" --labels-dates "$labels_date" --sc-analytics-tables all --user-filter-option unfiltered --sample-table-option sample --n-iter 10 --sequence-type shortlist --baseline-model-names "${baselines}" --transformer-model-names "${model_name}" --tokenizer-paths "${tokenizer_path}" --transformer-model-paths "${model_path}" --sgns-model-path ${sgns_model_path}  --bucket-name umap-user-model --n-class $n_class --output-path-local ${output}  --negative-label "<=15s" --max-model-input-size ${max_len} --run-name ${run_name} --reverse-sequence
    done
done
