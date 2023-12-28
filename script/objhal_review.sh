echo "========>Start evaluating answers<========"
review_file_name=hall_obj_halbench_answer_-1.json

base_dir=$1
answer_file_name=$2
coco_annotation_path=$3

python ./eval/eval_gpt_obj_halbench.py \
    --coco_path $coco_annotation_path \
    --cap_folder $base_dir \
    --cap_type $answer_file_name \
    --org_folder $q_file \
    --use_gpt \
    --openai_key $4 #'sk-TKMTtA1kmTEKNLvZ4d5bC6Fe934749A5B370Fb71037253A9'

python ./eval/summarize_gpt_obj_halbench_review.py $base_dir > $base_dir/obj_halbench_scores.txt

# Print Log
echo Scores are:
cat $base_dir/obj_halbench_scores.txt
echo done