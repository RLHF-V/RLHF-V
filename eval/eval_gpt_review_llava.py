import argparse
import json
import os
import time
import pathlib

from gpt4_grpc import get_eval

def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument('--openai_apikey', type=str, help='API_KEY for your OpenAI account.')

    args = parser.parse_args()

    chat = 'gpt-4-0314'

    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))


    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}


    reviewed_lines = []
    if pathlib.Path(args.output).exists():
        reviewed_lines = open(args.output).readlines()[:-1]
        print(f'Resume {args.output} from {len(reviewed_lines)}')
    review_file = open(f'{args.output}', 'w')
    if reviewed_lines:
        for line in reviewed_lines:
            review_file.write(line)
            review_file.flush()

    js_list = []
    handles = []
    for line_idx, (ques_js, ans1_js, ans2_js) in enumerate(zip(f_q, f_ans1, f_ans2)):
        if line_idx < len(reviewed_lines):
            continue
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]
        cap_str = '\n'.join(inst['captions'])
        box_str = '\n'.join([f'{instance["category"]}: {instance["bbox"]}' for instance in inst['instances']])

        category = json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n{box_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        output = {
            'id': line_idx,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', 0),
            'category': category}

        review = get_eval(chat, content, max_tokens=180, openai_apikey=args.openai_apikey)
        scores = parse_score(review)
        output['content'] = review
        output['tuple'] = scores
        review_file.write(json.dumps(output) + '\n')
        review_file.flush()

        # To avoid the rate limit set by OpenAI
        time.sleep(1)

    review_file.close()
