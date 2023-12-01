import json
import os
from collections import defaultdict

import numpy as np
import sys
import glob


if __name__ == '__main__':
    base_dir = sys.argv[1]
    print(base_dir)

    patterns = ['*', '*/*', '*/*/*']
    f_list = sum([list(glob.glob(os.path.join(base_dir, p))) for p in patterns], [])
    review_files = [x for x in f_list if x.endswith('.jsonl') and 'llava_test_gpt4' in x]

    for review_file in sorted(review_files):
        config = review_file.replace('gpt4_text_', '').replace('.jsonl', '')
        scores = defaultdict(list)
        print(f'GPT-4 vs. {config} #{len(list(open(review_file)))}')
        with open(review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                # filter failed case
                if review['tuple'][0] == -1:
                    print(f'#### Skip fail Case')
                    continue

                scores[review['category']].append(review['tuple'])
                scores['all'].append(review['tuple'])
        for k, v in scores.items():
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            print(k, stats, round(stats[1]/stats[0]*100, 1))
        print('=================================')
