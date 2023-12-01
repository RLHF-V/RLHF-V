###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===


base_dir=$1
to_process_ckpt_list="$1 "
save_dir=$2
# to_process_ckpt_list+=" $base_dir/checkpoint-40 $base_dir/checkpoint-80 $base_dir/checkpoint-120 $base_dir/checkpoint-160"
# to_process_ckpt_list+=" $base_dir/checkpoint-200 $base_dir/checkpoint-600 $base_dir/checkpoint-1000 $base_dir/checkpoint-1400 $base_dir/checkpoint-1800 $base_dir/checkpoint-2200 $base_dir/checkpoint-2600 $base_dir/checkpoint-3000"
# to_process_ckpt_list+=" $base_dir/checkpoint-400 $base_dir/checkpoint-800 $base_dir/checkpoint-1200 $base_dir/checkpoint-1600 $base_dir/checkpoint-2000 $base_dir/checkpoint-2400 $base_dir/checkpoint-2800 $base_dir/checkpoint-3200"
# to_process_ckpt_list+=" $base_dir/checkpoint-3600 $base_dir/checkpoint-4000 $base_dir/checkpoint-4400 $base_dir/checkpoint-4800 $base_dir/checkpoint-5200 $base_dir/checkpoint-5600 $base_dir/checkpoint-6000 $base_dir/checkpoint-6400"

# ===========> LLaVA Test Set <==============

answer_file_name="llava_test_answer.jsonl"

filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $save_dir/$answer_file_name
    if [[ ! -f $save_dir/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"


C=2
q_file=./eval/data/llava-bench_questions_with_image.jsonl

for ckpt_path in $filered_to_process_ckpt_list;
do
    answer_file=$save_dir/$answer_file_name
    echo "PWD at `pwd` checkpoint: "$ckpt_path" output to: "$answer_file

    echo "Start generating answers for $ckpt_path"
    CUDA_VISIBLE_DEVICES=$C python ./eval/wrap_muffin_vqa.py \
        --model-name $ckpt_path \
        --question-file $q_file \
        --answers-file  $answer_file &
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi
done
wait


echo "========>Start GPT4 Evaluating<========"
bash ./script/eval/batch_gpt4_review_llavabench.sh $save_dir $3
python ./eval/summarize_gpt_llava_review.py $save_dir >> $save_dir/llava_test_scores.txt

# Print Log
echo Scores are:
cat $save_dir/llava_test_scores.txt
echo done
