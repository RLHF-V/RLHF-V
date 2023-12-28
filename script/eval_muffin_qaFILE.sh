###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

to_process_ckpt_list="$1 "


q_file=$2
answer_file_name=$3

filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $ckpt/$answer_file_name
    if [[ ! -f $ckpt/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"


C=0

for ckpt_path in $filered_to_process_ckpt_list;
do
    answer_file=$ckpt_path/$answer_file_name
    echo "PWD at `pwd` checkpoint: "$ckpt_path" output to: "$answer_file

    CUDA_VISIBLE_DEVICES=$C python ./inference/muffin_vqa.py \
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
echo "========>Done generating answers<========"

