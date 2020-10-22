import argparse
import numpy as np
from matplotlib import pyplot as plt

feat_dim = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list_path', type=str, default='6000_extra_gallary_list.txt'.format(feat_dim))
    parser.add_argument('-f', '--features_path', type=str, default='6000_extra_gallary_{}_testerror'.format(feat_dim))
    parser.add_argument('-s', '--save_path', type=str, default='3000_extra_gallary_{}.png'.format(feat_dim))
    return parser.parse_args()

def get_scores(gallary, query):
    gallary = gallary / np.linalg.norm(gallary, axis=1, keepdims=True)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    scores = np.dot(gallary, query.T)
    scores = (scores + 1.) / 2.
    return scores

def plot(precisions, recalls, thrs, draw_idxes):
    plt.figure()
    plt.plot(precisions, recalls)
    plt.xlim(0.9975, 1.)
    plt.ylim(0.98, 1.)
    plt.xlabel('precision')
    plt.ylabel('recall')
    plt.grid(linestyle='-.')
    for i in draw_idxes:
        plt.text(precisions[i], recalls[i], '{:.3f} {:.3f} {:.3f}'.format(thrs[i], precisions[i], recalls[i]), ha='right', va='bottom', fontsize=6)

    # cur_pre = precisions[0]
    # plt.text(cur_pre, recalls[0], '{},{},{}'.format(thrs[0], cur_pre, recalls[0]))
    # for i, pre in enumerate(precisions):
    #     if pre != cur_pre:
    #         plt.text(pre, recalls[i], '{}, {}, {}'.format(thrs[i], pre, recalls[i]))
    plt.savefig(args.save_path)

args = parse_args()
gallary = []
query = []
labels = []
with open(args.list_path) as f:
    for line in f.readlines():
        labels.append(int(line.strip().split()[-1]))
with open(args.features_path) as f:
    for i, line in enumerate(f.readlines()):
        if labels[i] == 0:
            gallary.append([float(x) for x in line.strip().split()])
        elif labels[i] == 1:
            query.append([float(x) for x in line.strip().split()])

gallary = np.array(gallary, dtype=np.float32)
query = np.array(query, dtype=np.float32)
scores = get_scores(gallary, query)
top1_idxes = np.argmax(scores, axis=0)
top1_scores = scores[top1_idxes, np.array(range(len(query)))]
for i, x in enumerate(top1_idxes):
    if x != i:
        print('i={}, x={}, scores={}'.format(i, x, top1_scores[i]))
print('top1_scores={}'.format(top1_scores))
min_score, max_score = top1_scores.min(), top1_scores.max()
print('min_score={}'.format(min_score))
print('max_score={}'.format(max_score))
step = (max_score - min_score) / 100
precisions = []
recalls = []
thrs = []
tmp_fp = -1
draw_idxes = []
for i in range(100):
    thr = min_score + step * i
    thrs.append(thr)
    valids = top1_scores > thr
    labels = np.array(range(len(query)))
    tp = float(np.sum((top1_idxes == labels) & valids))
    fp = float(np.sum((top1_idxes != labels) & valids))
    tn = 0.
    fn = float(np.sum(1 - valids))
    precisions.append(tp / (tp + fp))
    if fp != tmp_fp:
        draw_idxes.append(i)
        tmp_fp = fp
    recalls.append(tp / (tp + fn))
print('draw_idxes={}'.format(draw_idxes))
plot(precisions, recalls, thrs, draw_idxes)

