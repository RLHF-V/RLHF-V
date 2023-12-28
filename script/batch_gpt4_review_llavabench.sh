SOURCE_DIR=$1
limit=100
C=0
process_limit=2
force=no_force

while IFS= read -r -d '' -u 9
do
    # echo "reply "$REPLY
    if [[ $REPLY == *$prefix*llava_test_answer.jsonl ]]; then
        if [[ $force == force ]]; then
            rm -f $REPLY.llava_test_gpt4.jsonl
        fi
        echo "EVAL qa90 "$REPLY
        python ./eval/eval_gpt_review_llava.py \
            --question ./eval/data/qa90_questions.jsonl \
            --context ./eval/data/caps_boxes_coco2014_val_80.jsonl \
            --answer-list \
            ./eval/data/qa90_gpt4_answer.jsonl \
            $REPLY \
            --rule ./eval/data/rule.jsonfile \
            --output $REPLY.llava_test_gpt4.jsonl \
            --openai_apikey $2 &
        sleep 5

        C=$((C+1))
        echo "C=$C"
        if [[ $C == $process_limit ]]; then
            echo "Wait for next iteration"
            C=0
            wait
        fi
    fi
done 9< <( find $SOURCE_DIR -type f -name "*llava*" -exec printf '%s\0' {} + )

wait