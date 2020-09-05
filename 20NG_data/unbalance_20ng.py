import random
from collections import defaultdict

random.seed(42)

files = ['train.tsv' , 'test.tsv']
ratio = 0.1
unbalance_ratio = 1 - ratio  #   number of tail class / number of head class
total_class = 20
for file in files:
    data = defaultdict(list)
    with open(file, 'r') as f:
        for l in f:
            text, label = l.split("\t")
            data[label.strip()].append(text)
    # sort data by its examples num
    print(data.keys())
    sort_data = sorted(data.items(), key=lambda item: len(item[1]), reverse=True)  # sort by number
    max_class = max([len(x[1]) for x in sort_data])
    unbalanced_data = defaultdict(list)
    for i, item in enumerate(sort_data):
        random.shuffle(item[1])  # shuffle
        left = max_class * (1 - i * unbalance_ratio / (total_class - 1))
        selected = item[1][:int(left)]
        unbalanced_data[item[0]].extend(selected)
    # print(len(unbalanced_data))
    cnt = 0
    with open(file + '.unbal%.3f' %  ratio, 'w') as fw:
        for k in unbalanced_data:
            for line in unbalanced_data[k]:
                fw.write('%s\t%s\n' % (line, k))
                cnt += 1
    print('total num: %d' % cnt )
