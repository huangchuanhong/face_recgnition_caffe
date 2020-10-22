# encoding=utf-8

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os, datetime, random, math
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

savepicname = 'compare_1:1'  # 'Save pic name'
numpoint = 5000
fplist = './fplist'
fp_point = [1e-5, 1e-4, 1e-3, 1e-2]
draw = 1
write = 0  # 'whether to save fp list, default is 1: 1 is ON, 0 is OFF'
m = 5


def getlabels(file, mode):
    lines = open(file).read().strip().split()
    #lines = open(file).read().replace("\n", " ").split(" ")[:-1]
    size = len(lines)
    labels = np.array(lines).reshape((-1, 3))
    gallery = np.where(labels[:, 2] == "0")[0]
    query = np.where(labels[:, 2] == "1")[0]
    dic = todict(labels[gallery][:, 1])
    vfunc = np.vectorize(checkexists)
    if mode == 0:
        GroundTruth = vfunc(labels[query][:, 1], dic)
        print('GroundTruth.size:', len(GroundTruth), ' GroundTruth.sum:', GroundTruth.sum())
        print("gallery:", len(gallery), " query:", len(query))
        return labels, gallery, query, GroundTruth
    else:
        print("No mode: ", mode)


def checkexists(v, dic):
    return dic.__contains__(v)


def todict(array):
    return dict(zip(array, [1] * len(array)))


def readFeatures(filename, size, dtype="cos"):
    if filename[-3:] == 'npz':
        try:
            npzfiles = np.load(filename)
            feats = npzfiles["feats"]
            feat_dim = npzfiles["feat_dim"]
            print("Feature dim: %d" % feat_dim)
            feats = normalize(feats) if dtype == "cos" else feats
            return feats, feat_dim
        except:
            print("Need to update npz file for newer version")
            sys.exit()
    flines = open(filename).read()
    fs = flines.strip().split()#[:-1]
    #fs = flines.replace("\n", " ").split(" ")#[:-1]
    print(len(fs))
    feats = np.array(fs, dtype='f')
    feat_dim = int(1.0 * feats.shape[0] / size)
    print("Feature dim: %d" % feat_dim)
    feats = feats[:size * feat_dim, ...].reshape((size, -1))
    feats = normalize(feats) if dtype == "cos" else feats
    return feats, feat_dim


def score(dist):
    s = (1 + dist) / 2
    return s


def getscores(feat_list, GroundTruth, labels, gallery, query, features, feat_dim, mode):
    feat_Q = features[query]
    feat_G = features[gallery]
    querys = labels[query]
    gallerys = labels[gallery]
    MissedG = np.where(np.sum(feat_G * feat_G, axis=1) == 0)
    MissedQ = np.where(np.sum(feat_Q * feat_Q, axis=1) == 0)

    # print("feature read")
    if mode == 0:
        savename = feat_list + '.npz'

        if os.path.exists(savename):
            print("load data from", savename)
            y_true, cos_matrix, querys, gallerys, shape, f_dim = loadData(savename)
            if shape[1] == len(gallerys) and feat_dim == f_dim:
                return y_true, cos_matrix, querys, gallerys
            else:
                print(".npz file has no match features shape")
                # sys.exit()
        # cosine 
        #print("Calculate cosine similarity...")
        cos_matrix = cosine_similarity(feat_Q, feat_G)
        #print('Calculate cosine similarity end.')
        #cos_matrix = pairwise_distances(feat_Q, feat_Q, metric='euclidean', n_jobs=-2)
        #dis_matrix = euclidean_distances(feat_Q,feat_G)
        #dis_matrix = dis_matrix**2
        #score_max = 0.9888888
        #score_min = 0.010000
        #a = (np.log(1.0/score_max - 1)-np.log(1.0/score_min-1))/(dis_matrix.min()-dis_matrix.max())
        #b = np.log(1.0/score_min - 1)-a*dis_matrix.max()
        #cos_matrix = 1.0/(1.0+np.exp(a*dis_matrix+b))
        #print(cos_matrix[:,0])
        shape = np.shape(cos_matrix)
        cos_matrix = score(cos_matrix)
        y_true = np.zeros(shape=[len(query), len(gallery)])
        for i in range(len(gallerys)):
            t = np.where(querys[:, 1] == gallerys[:, 1][i], 1, 0)
            y_true[:, i] = t

        saveData(savename, y_true, cos_matrix, querys, gallerys, shape, feat_dim)
        return y_true, cos_matrix, querys, gallerys
    else:
        print("No mode: ", mode)


def getStyle(cnt):
    color = ["r", "b", "g", "c", "y", "k", "m"]
    marker = ["o", "v", "s", "x", "+", "."]
    linestyle = ["", "--", ":"]
    return color[cnt % 7] + linestyle[cnt % 3]


def plot(feat_list, query, y_true, cos_matrix, querys, gallerys, GroundTruth, numpoint, style):
    query = np.array(query)
    y_true = np.array(y_true).astype(bool)
    min_y = cos_matrix.min()
    max_y = cos_matrix.max()
    print("max_score=", max_y, " min_score=", min_y)
    print("y_true.size()=", len(y_true), " y_true.sum()=", y_true.sum())
    step = 1.0 * (max_y - min_y) / numpoint

    far = []
    recall = []
    Threshold = []
    fppoint = fp_point

    i = 0
    while i <= numpoint:
        thre = max_y - i * step
        i += 1
        tmp = cos_matrix >= thre
        fp = tmp & (~ y_true)
        if fp.sum() == 0:
            continue
        # fn = (~tmp) & (y_true)
        tp = tmp & y_true
        tn = (~tmp) & (~y_true)
        recall_ = 1.0 * tp.sum() / y_true.sum()
        far_ = 1.0 * fp.sum() / (fp.sum() + tn.sum())
        #far_ = 1.0 * fp.sum() / (fp.sum() + tp.sum())
        # print('i=', i, ' fn =', fn.sum(), 'tp =', tp.sum(), 'fp =', fp.sum(), 'tn =', tn.sum(), 'far=', far_, 'rec=',
        #       recall_, 'thre=', thre)
        if fp.sum() >= tp.sum() or thre <= 0.01:
            i = i + int(i / m)
            if i > numpoint:
                i == numpoint
        if write:
            if not os.path.exists(fplist):
                os.mkdir(fplist)
            if fppoint != [] and far_ >= fppoint[0]:
                fppath = os.path.join(fplist, str(fppoint[0]))
                filename = os.path.join(fppath, 'fplist_' + str(fppoint[0]) + '_' + str(round(thre, 5)) + '.txt')
                if not os.path.exists(fppath):
                    os.mkdir(fppath)
                index = np.where(fp == 1)
                row = index[0]
                col = index[1]
                with open(filename, 'w') as f:
                    for j in range(len(row)):
                        cont = str(query[row[j]]) + ' ' + querys[row[j]][0] + ' ' + querys[row[j]][1] + ' ' + \
                               gallerys[col[j]][0] + ' ' + \
                               gallerys[col[j]][1] + ' ' + str(cos_matrix[row[j], col[j]]) + ' ' + str(
                            int(GroundTruth[j])) + '\n'
                        f.write(cont)
                    f.close()
                print(filename, 'has saved')
                exit()
                fppoint = fppoint[1:]

        if fp.sum() == 0:
            continue
        else:
            far.append(far_)
            recall.append(recall_)
            Threshold.append(thre)
        if thre == min_y:
            break

    if True:
        np.savez(str(feat_list) + '_far_rec_thr.npz', far=far, recall=recall, Threshold=Threshold)
        print(str(feat_list) + '_far_rec_thr.npz', 'has saved')

    if draw:
        plotdraw(far, recall, Threshold, style)


def plotdraw(far, recall, Threshold, style):
    minx = 1e-6
    plt.plot(far, recall, style)
    plt.xscale("log")
    #plt.xlim([10 ** int(math.log(minx, 10) - 1), 10 ** int(math.log(1e-3, 10) - 1)])
    plt.xlim(0,1.01)
    #plt.xlim([10 ** int(math.log(minx, 10) - 1), 1.5])
    plt.ylim(0.,1.01)
    #plt.xlabel('FAR')
    #plt.ylabel('RECALL')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('1:1')
    plt.grid(linestyle='-.')

    find = min(far)
    if max(far) == 1:
        x = 0
    else:
        x = 1
    for i in range(x, int(-math.log(find, 9)) + 1):
        milli = np.searchsorted(far, 0.1 ** i, side='left')
        try:
            f = far[milli]
            r = recall[milli]
            t = Threshold[milli]
            # print(r, t)
            # print("%.e" % (0.1 ** i))
            plt.text(0.1 ** i, r, ("%.e" % (0.1 ** i), round(r, 5)), ha='right', va='bottom', fontsize=6)
            plt.text(0.1 ** i, r, round(t, 6), ha='right', va='top', fontsize=6)
            plt.scatter(0.1 ** i, r, c='k', marker='.')
        except IndexError:
            continue


def getsavename(sn, suff):
    name = sn
    suffix = 0
    while os.path.exists(name + "_" + str(suffix) + suff):
        suffix += 1
    savename = name + "_" + str(suffix) + suff
    return savename


def saveData(filename, y_true, cos_matrix, querys, gallerys, shape, feat_dim):
    np.savez(filename, y_true=y_true, cos_matrix=cos_matrix, querys=querys, gallerys=gallerys, shape=shape,
             feat_dim=feat_dim)


def loadData(filename):
    try:
        npzfiles = np.load(filename)
        y_true = npzfiles["y_true"]
        cos_matrix = npzfiles["cos_matrix"]
        querys = npzfiles["querys"]
        gallerys = npzfiles["gallerys"]
        shape = npzfiles["shape"]
        f_dim = npzfiles["feat_dim"]
        return y_true, cos_matrix, querys, gallerys, shape, f_dim
    except:
        print("Need to update npz file for newer version")
        sys.exit()


def checksavedata():
    return


if __name__ == '__main__':
    feat_list = sys.argv[2:]
    mode = 0
    label_list = sys.argv[1:2][0]
    kind = len(feat_list)
    print('label:', label_list, ' feat:', feat_list)
    plt.figure()

    labels, gallery, query, GroundTruth = getlabels(label_list, mode)
    for i in range(kind):
        print('start', feat_list[i], '...')
        features, feat_dim = readFeatures(feat_list[i], len(labels))

        if os.path.exists(str(feat_list[i]) + '_far_rec_thr.npz'):
            npzfiles = np.load(str(feat_list[i]) + '_far_rec_thr.npz')
            print('load data from', str(feat_list[i]) + '_far_rec_thr.npz')
            plotdraw(npzfiles['far'], npzfiles['recall'], npzfiles['Threshold'], getStyle(i))
        else:
            y_true, cos_matrix, querys, gallerys = getscores(feat_list[i], GroundTruth, labels, gallery, query,
                                                             features,
                                                             feat_dim, mode)
            plot(feat_list[i], query, y_true, cos_matrix, querys, gallerys, GroundTruth, numpoint, getStyle(i))

    plt.legend(labels=feat_list, fontsize='x-small', loc=0)
    savename = getsavename(savepicname, '.png')
    plt.savefig(savename, dpi=200)
    plt.show()
    print(savename, 'has saved')
    print('The end.')
