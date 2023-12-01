###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

base_dir=$1
to_process_ckpt_list="$1 "
save_dir=$2


q_file=./eval/data/obj_halbench_300_with_image.jsonl
answer_file_name=obj_halbench_answer.jsonl

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

C=0

for ckpt_path in $filered_to_process_ckpt_list;
do
    if [[ ! -f $save_dir/$answer_file_name ]]; then
        answer_file=$save_dir/$answer_file_name
        echo "PWD at `pwd` checkpoint: "$ckpt_path" do Object HalBench output to: "$answer_file

        CUDA_VISIBLE_DEVICES=$C python ./eval/wrap_muffin_vqa.py \
            --model-name $ckpt_path \
            --question-file $q_file \
            --answers-file $answer_file &
        C=$((C+1))
        echo "C=$C"
        if [[ $C == 8 ]]; then
            echo "Wait for next iteration"
            C=0
            wait
        fi
    fi

done
wait

echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"
review_file_name=hall_obj_halbench_answer_-1.json
coco_annotation_path=$3

python ./eval/eval_gpt_obj_halbench.py \
    --coco_path $coco_annotation_path \
    --cap_folder $save_dir \
    --cap_type $answer_file_name \
    --org_folder $q_file \
    --use_gpt \
    --openai_key $4 #'sk-TKMTtA1kmTEKNLvZ4d5bC6Fe934749A5B370Fb71037253A9'

python ./eval/summarize_gpt_obj_halbench_review.py $save_dir > $save_dir/obj_halbench_scores.txt

# Print Log
echo Scores are:
cat $save_dir/obj_halbench_scores.txt
echo done
