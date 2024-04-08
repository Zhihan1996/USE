
export batch_size=8
export min_l=128
export max_l=512
export epoch=10
export max_file=5000
export gap=50
export learning_rate=4e-4
export gradient_accumulation_steps=8
export alibi=False
export time=False
export model=retnet
export num_heads=8
export contrastive_embedding=token
export contrastive_type=Orig
export obj=CLM
export fep=686
export fep_context_length=0
export fep_loss_frequency=1
export run_name=retnet_clm

python train_user_model.py \
            --sc-analytics-tables all \
            --bucket-name umap-user-model \
            --sample-n 1000000 \
            --run-name ${run_name} \
            --features-start-date 20230401 \
            --features-end-date 20230414 \
            --sequence-type shortlist \
            --user-filter-option filtered \
            --n-iter 1 \
            --num-train-epochs ${epoch} \
            --max-files ${max_file} \
            --min-model-input-size ${min_l} \
            --max-model-input-size ${max_l} \
            --user-segments-gap ${gap} \
            --per-device-batch-size ${batch_size} \
            --test-size 0.05 \
            --warmup-ratio 0.06 \
            --learning-rate ${learning_rate} \
            --gradient-accumulation-steps ${gradient_accumulation_steps} \
            --tokenizer-path artifacts/tokenizers/bigbird_word \
            --model-output-path ../artifacts/${run_name} \
            --eval-steps 500 \
            --model ${model} \
            --reverse-sequence \
            --num-heads ${num_heads} \
            --contrastive-embedding ${contrastive_embedding} \
            --contrastive-type ${contrastive_type} \
            --fep-event-of-interests event_of_interests_${fep} \
            --fep-context-length ${fep_context_length} \
            --fep-loss-frequency ${fep_loss_frequency} \
            --get-random-segment \
            --training-objective ${obj}
