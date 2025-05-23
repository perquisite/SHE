import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.numpy()

def multi_test(test_data, test_labels, retrieval_data, retrieval_labels,  k =-1, metric='cosine'):
    n_view = len(test_data)
    res = np.zeros([n_view, n_view])
    for i in range(n_view):
        for j in range(n_view):
            if i == j:
                continue
            else:
                if len(test_labels[j].shape) == 1:
                    tmp = fx_calc_map_label(retrieval_data[j], retrieval_labels[j], test_data[i], test_labels[i], k, metric=metric)
                else:
                    tmp = fx_calc_map_multilabel(retrieval_data[j], retrieval_labels[j], test_data[i], test_labels[i], k, metric=metric)
                res[i, j] = tmp
    return res

import scipy
def fx_calc_map_label(dbase, dbase_labels, test, test_label, k = -1, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, dbase, metric)

    ord = dist.argsort(1)
    if k == -1:
        k = dbase_labels.shape[0]
    def calMAP(rek):
        ap = []
        for i in range(len(test_label)):
            order = ord[i]
            pre = []
            r = 0.0
            for j in range(rek):
                if test_label[i] == dbase_labels[order[j]]:
                    r += 1.
                    pre.append(r / (float(j) + 1.))
            if r > 0:
                ap += [np.sum(pre) / r]
            else:
                ap += [0]

        return np.mean(ap)
    res = calMAP(k)
    return res

def fx_calc_map_multilabel(dbase, dbase_labels, test, test_label, k=-1, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, dbase, metric)
    ord = dist.argsort()
    numcases = dist.shape[1]
    if k == -1:
        k = numcases
    res = []
    for i in range(dist.shape[0]):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(dbase_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def ind2vec(ind, N=None):
    if len(ind.shape) == 1:
        ind = ind.reshape([-1,1])
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() - ind.min() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)
def predict(model, data, p_emb_inputs, batch_size=32):
    batch_count = int(np.ceil(data.shape[0] / float(batch_size)))
    results = []
    with torch.no_grad():
        for i in range(batch_count):
            if torch.is_tensor(data):
                batch = data[i * batch_size: (i + 1) * batch_size]
            else:
                batch = (torch.tensor(data[i * batch_size: (i + 1) * batch_size])).cuda()
            results.append(to_data(model(batch, p_emb_inputs)[0]))
    return np.concatenate(results)
def get_pr(qB, rB, label):
        label = torch.tensor(ind2vec(label).astype(int), requires_grad=False)
        num_query = qB.shape[0]
        topK = rB.shape[0]
        # topK =50
        P, R = [], []
        dist = scipy.spatial.distance.cdist(qB, rB, 'cosine')
        Rank = np.argsort(dist)
        Gnd = (label.mm(label.transpose(0, 1)) > 0).type(torch.float32)
        for k in range(1, topK + 1):  # 枚举 top-K 之 K  
            p = np.zeros(num_query)    
            r = np.zeros(num_query)  
            for it in range(num_query):
                gnd = Gnd[it]
                gnd_all = gnd.sum()  # 整个被检索数据库中的相关样本数
                if gnd_all == 0:
                    continue
                asc_id = Rank[it][:k]
                gnd = gnd[asc_id]
                gnd_r = gnd.sum()  # top-K 中的相关样本数
                p[it] = gnd_r / k  
                r[it] = gnd_r / gnd_all  
    
            P.append(np.mean(p))
            R.append(np.mean(r))
        S = np.arange(topK)
        S = S.tolist()
        return P, R, S
def pr_curve(P, R, filename):
    fig = plt.figure(figsize=(5, 5))
    # plt.grid(linestyle = "--") #设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False) #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    #markevery为间隔点、marker为点的形式、linestyle为线的形式,都可选
    labels = ['a']
    colors = ['r']
    markers = ['*']
    linestyles = ['-']
    for i in range(len(P)):
        plt.plot(R[i], P[i], color=colors[i],label=labels[i],linewidth=1.5,linestyle=linestyles[i], marker=markers[i], markevery=10)
    # plt.plot(b_i2t_r, b_i2t_p,color="lightgreen",label="b",linewidth=1.5, linestyle="--", marker='*', markevery=270)
    plt.grid(True)
    plt.xlim(0, 1)#x轴范围，可调整
    plt.ylim(0, 1)#y轴范围，可调整
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.title("Image2Text",fontsize=12,fontweight='bold') #默认字体大小为12
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10,fontweight='bold') #设置图例字体的大小和粗细
    plt.savefig(filename)
def get_relation_score(model, img_embs, cap_embs, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            # sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                sim = model(im, ca) 

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy().reshape([im_end - im_start, -1])
    return sims

