
# Declare an array of string with type
echo "strat"
declare -a models=("/home/nlp/shaked571/ParaShoot/alephbert-base" "bert-base-multilingual-cased" "avichr/heBERT")
for i in 1 2 3 4 5
do
	for model in "${models[@]}";
	do
       	echo "Output:"
       	echo "model nameo r path: ${model}"
        ModelName=$(basename "$model")

        echo "Model Name: ${ModelName}"
       	python -u "/home/nlp/shaked571/ParaShoot/run_qa.py"  \
       	--train_file "/home/nlp/shaked571/ParaShoot/data/train.json" \
        --validation_file "/home/nlp/shaked571/ParaShoot/data/dev.json" \
        --test_file "/home/nlp/shaked571/ParaShoot/data/test.json" \
        --model_name_or_path "${model}" \
        --output_dir "/home/nlp/shaked571/ParaShoot/${ModelName}/${i}" \
        --max_answer_length 50 \
        --version_2_with_negative false \
        --num_train_epochs 15 \
        --per_device_train_batch_size 8 \
        --save_steps 12000 \
        --n_best_size 10 \
        --eval_steps 30 \
        --do_train \
        --do_eval \
        --do_predict \
        --fp16 \
        --overwrite_output_dir \
        --seed ${i} \
        --warmup_steps 100 \
        --logging_first_step \
        --greater_is_better true

	done
done
