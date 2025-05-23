import torch
import numpy as np
from torch import optim
import cal_utils as utils
import torch.nn.functional as F
import data_loader
import scipy.io as sio
import copy
import time
import cal_utils as utils
def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    Sim[Sim>0] = 1
    return Sim
def ce_loss(pred, labels):
    loss = - (labels * pred.log()).sum(1).mean()
    return loss
def gce_loss(pred, labels, q=0.01):
    gae = (1. - torch.sum(labels.float() * pred, dim=1)**q).div(q)
    return gae.mean()
def dhl_loss(data_fea, labels, tau, q = 0.7):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
    sim = calc_label_sim(labels, labels)
    S_ = cos(data_fea, data_fea)
    P_ = ((S_ * sim) / tau).exp()
    N_ = ((S_ * (1 - sim)) / tau).exp()
    tmp = P_.sum(1) / (P_ +  N_).sum(1)
    loss = - tmp.log().mean()
    return loss
# def l_loss(sims, labels, tau = 1, q = 0.7): # 所有代理的相似度之和计算损失
#     sims = F.softmax(sims / tau, dim=1)
#     repeat_num = sims.shape[1] / labels.shape[1]
#     labels = labels.repeat_interleave(int(repeat_num), dim=1)

#     # 计算同类样本的相似度之和
#     same_class_sims = sims * labels 
    
#     loss = - (same_class_sims.sum(1)).log()
#     return loss.mean()
def klm_loss(sims, labels, tau = 1, q = 0.7): # 最近代理计算损失
    sims = (sims+1)/2. # 将相似度转为0-1之间
    repeat_num = sims.shape[1] / labels.shape[1]
    labels = labels.repeat_interleave(int(repeat_num), dim=1)

    # 计算同类样本的相似度之和
    same_class_sims = sims * labels 
    max_class_sims, _ = torch.max(same_class_sims, dim=1, keepdim=True)
    
    loss = - (max_class_sims.sum(1)).log()
    return loss.mean()

# def klm_loss_all(sims, labels, tau = 1, q = 0.7): # 最近代理计算损失
#     sims = (sims+1)/2. # 将相似度转为0-1之间
#     repeat_num = sims.shape[1] / labels.shape[1]
#     labels = labels.repeat_interleave(int(repeat_num), dim=1)

#     # 计算同类样本的相似度之和
#     same_class_sims = sims * labels
#     mean_same_class_sims = same_class_sims.sum(1) / repeat_num
#     # max_class_sims, _ = torch.max(same_class_sims, dim=1, keepdim=True)
    
#     loss = - (mean_same_class_sims).log()
#     return loss.mean()

def klm_score(sims, labels):
    sims = (sims+1)/2. # 将相似度转为0-1之间
    repeat_num = sims.shape[1] / labels.shape[1]
    labels = labels.repeat_interleave(int(repeat_num), dim=1)

    # 计算同类样本的相似度之和
    same_class_sims = sims * labels 
    max_class_sims, _ = torch.max(same_class_sims, dim=1, keepdim=True)
    
    score = max_class_sims.sum(1)
    return score.mean()

def klr_loss(prototype_vectors, num_class, tau=1):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
    num_prototype = prototype_vectors.shape[0]
    num_prototype_per_class = int(num_prototype / num_class)
    labels = torch.zeros(num_prototype, num_class).cuda()
    for i in range(num_class):
        labels[i * num_prototype_per_class: (i+1) * num_prototype_per_class, i] = 1
    sim = calc_label_sim(labels, labels)
    S_ = cos(prototype_vectors, prototype_vectors)
    S_ = S_ - S_.diag().diag()
    P_ = ((S_ * sim) / tau).exp()
    N_ = ((S_ * (1 - sim)) / tau).exp()
    tmp = P_.sum(1) / (P_ + N_).sum(1)
    loss = - tmp.log().mean()
    return loss
def klr_orthogonal_loss(prototype_vectors, num_class, tau=1):

    similarity_matrix = torch.mm(prototype_vectors, prototype_vectors.t())
    num_prototypes = prototype_vectors.shape[0]
    num_prototypes_per_class = int(num_prototypes / num_class)
    # 生成掩码矩阵，只保留不同类别的原型点积
    mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    for i in range(num_class):
        mask[i * num_prototypes_per_class : (i + 1) * num_prototypes_per_class,
             i * num_prototypes_per_class : (i + 1) * num_prototypes_per_class] = False  # 同类别不约束

    # 计算正交损失
    loss = torch.mean(torch.abs(similarity_matrix[mask]))  # 让不同类别的点积趋近于 0
    return loss


def klr_score(prototype_vectors, num_class, tau=1):
    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
    num_prototype = prototype_vectors.shape[0]
    num_prototype_per_class = int(num_prototype / num_class)
    labels = torch.zeros(num_prototype, num_class).cuda()
    for i in range(num_class):
        labels[i * num_prototype_per_class: (i+1) * num_prototype_per_class, i] = 1
    sim = calc_label_sim(labels, labels)
    S_ = cos(prototype_vectors, prototype_vectors)
    S_ = S_ - S_.diag().diag()
    P_ = ((S_ * sim) / tau).exp()
    N_ = ((S_ * (1 - sim)) / tau).exp()
    tmp = P_.sum(1) / (P_ + N_).sum(1)
    score = tmp.mean()
    return score


def klt_loss(prototype_vectors, prototype_vectors_anchor, bound, tau=1):
    sim = F.cosine_similarity(prototype_vectors.unsqueeze(1), prototype_vectors_anchor.unsqueeze(0), dim=2)
    sim_diag = sim.diag()
    same_label_loss = -(torch.min(torch.tensor(1.0, device=sim_diag.device), F.relu(sim_diag - bound + 1))).clamp(min=1e-10).log().mean()
    return same_label_loss
class Solver(object):
    def __init__(self, config, logger):
        self.logger = logger
        self.bits = config.bits
        self.first_modality = config.first_modality
        data = data_loader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        (self.train_data, self.train_labels, self.test_data, self.test_labels, self.retrieval_data, self.retrieval_labels) = data

        self.n_view = len(self.train_data)
        train_index = list(range(self.n_view))
        self.train_index = [train_index[self.first_modality]] + train_index[:self.first_modality] + train_index[self.first_modality+1:]
        for v in range(self.n_view):
            if min(self.train_labels[v].shape) == 1:
                self.train_labels[v] = self.train_labels[v].reshape([-1])
            if min(self.retrieval_labels[v].shape) == 1:
                self.retrieval_labels[v] = self.retrieval_labels[v].reshape([-1])
            if min(self.test_labels[v].shape) == 1:
                self.test_labels[v] = self.test_labels[v].reshape([-1])

        if len(self.train_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.train_labels).reshape([-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.train_labels[0].shape[1]


        self.dropout_prob = 0.5
        self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eta = config.eta
        self.bound = config.bound
        self.q = config.q
        self.num_prototypes_per_class = config.num_prototypes_per_class
        self.num_prototypes = self.num_prototypes_per_class * self.num_classes
        self.prototype_vectors = torch.rand(self.num_prototypes, self.bits).tanh()
        # self.prototype_vectors = self.init_prototype_vectors(self.prototype_vectors, self.num_classes, self.num_prototypes_per_class)

        self.tau = config.tau
        self.view_id = config.view_id
        self.gpu_id = config.gpu_id
        self.epochs = config.epochs
        self.sample_interval = config.sample_interval
        self.just_val = config.just_val
        self.seed = config.seed
        self.config = config
        self.is_fine_tuning = config.is_fine_tuning

        print("datasets: %s, batch_size: %d, bits: %d, hyper-alpha1: %f "% (
                (config.datasets, self.batch_size, self.bits, self.alpha)))
            
        self.runing_time = config.running_time
        self.threshold = 0

    def to_one_hot(self, x):
        if len(x.shape) == 1 or x.shape[1] == 1:
            one_hot = (self.classes.reshape([1, -1]) == x.reshape([-1, 1])).astype('float32')
            labels = one_hot
            y = torch.tensor(labels).cuda()
        else:
            y = torch.tensor(x.astype('float32')).cuda()
        return y
    def init_pemb_input(self):
        inputs = torch.zeros(self.num_prototypes, self.num_classes).cuda()
        seed = 0
        for i in range(self.num_classes):
            for j in range(i * self.num_prototypes_per_class, (i+1) * self.num_prototypes_per_class):
                torch.manual_seed(seed)
                seed += 1
                inputs[j, i] = min(1 + 0.1 * torch.randn(1), 1)
        return inputs
    # def init_prototype_vectors(self, prototype_vectors, num_class, num_prototypes_per_class):
    #     C = num_class
    #     M = num_prototypes_per_class
    #     bits = prototype_vectors.shape[1]
    #     for i in range(C):
    #         for j in range(M):
    #             prototype_vectors[i * M + j] = prototype_vectors[i] + 0.01 * torch.randn(bits)  # 0.01 控制相似度
    #     return prototype_vectors

    def view_result(self, _acc):
        res = ''
        res += ((' - mean: %.5f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
        for _i in range(self.n_view):
            for _j in range(self.n_view):
                if _i != _j:
                    res += ('%.5f' % _acc[_i, _j]) + ','
        return res


    def train(self):
        if self.view_id >= 0:
            start = time.time()
            self.train_view(self.view_id)
        else:
            start = time.time()
            for v in self.train_index:
                    self.train_view(v)

        end = time.time()
        runing_time = end - start
        if self.runing_time:
            print('runing_time: ' + str(runing_time))
        test_fea, test_lab, retrieval_fea, retrieval_lab = [], [], [], []
        for v in range(self.n_view):
            tmp = sio.loadmat('features/' + self.datasets + '_' + str(v) + '_' + str(self.bits) + '.mat')
            test_fea.append(tmp['test_fea'])
            test_lab.append(tmp['test_lab'].reshape([-1,]) if min(tmp['test_lab'].shape) == 1 else tmp['test_lab'])
            retrieval_fea.append(tmp['retrieval_fea'])
            retrieval_lab.append(tmp['retrieval_lab'].reshape([-1,]) if min(tmp['retrieval_lab'].shape) == 1 else tmp['retrieval_lab'])
        
        test_results = utils.multi_test(test_fea, test_lab, retrieval_fea, retrieval_lab, metric='hamming')
        print("test resutls@all:" + self.view_result(test_results))
        self.logger.info("test resutls@all:" + self.view_result(test_results))
        # test_results = utils.multi_test(test_fea, test_lab, 50)
        # print("test resutls@50:" + self.view_result(test_results))
        # self.logger.info("test resutls@50:" + self.view_result(test_results))
        # sio.savemat('features/' + self.datasets + '_MARS_test_feature_results.mat', {'test': test_fea, 'test_labels': test_lab})
    def train_score(self):
        start = time.time()
        for v in self.train_index:
            self.evaluate_priors(v)
        end = time.time()
        runing_time = end - start
        if self.runing_time:
            print('runing_time: ' + str(runing_time))
            return runing_time
    def train_view(self, view_id):
        from to_seed import to_seed
        to_seed(seed=self.seed)
        import os
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        if view_id == self.first_modality:
            from model import Prototype_Net
            p_net = Prototype_Net(input_dim=self.input_shape[view_id],num_class=self.num_classes, 
                                  num_prototypes_per_class=self.num_prototypes_per_class, out_dim=self.bits).cuda()
            p_emb_inputs = self.init_pemb_input()
            get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
            params_pnet = get_grad_params(p_net)
            optimizer_pnet = optim.Adam(params_pnet, self.lr[view_id], [0.5, 0.999])
            best_acc = 0
            best_loss = 1e9
            best_epoch = 0
            for epoch in range(self.epochs):
                p_net.train()
                batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

                rand_didx = np.arange(self.train_data[view_id].shape[0])
                np.random.shuffle(rand_didx)

                for batch_idx in range(batch_nums):

                    didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                    view_labs = self.to_one_hot(self.train_labels[view_id][didx])
                    view_data = self.to_one_hot(self.train_data[view_id][didx])
                    optimizer_pnet.zero_grad()
                    data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                    loss1 = klm_loss(sims, view_labs, tau=self.tau)
                    # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                    loss2 = dhl_loss(data_fea, view_labs, self.tau)
                    loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                    # loss3 = klr_orthogonal_loss(prototype_vectors, self.num_classes, tau=self.tau)
                    loss = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3

                    loss.backward()
                    optimizer_pnet.step()

                    if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                        p_net.eval()
                        view_labs = self.to_one_hot(self.test_labels[view_id])
                        view_data = self.to_one_hot(self.test_data[view_id])
                        data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                        loss1 = klm_loss(sims, view_labs, tau=self.tau)
                        # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                        loss2 = dhl_loss(data_fea, view_labs, self.tau)
                        loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                        # loss3 = klr_orthogonal_loss(prototype_vectors, self.num_classes, tau=self.tau)
                        loss_test = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3


                        # test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                        # test_labs = self.test_labels[view_id]
                        # retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                        # retrieval_labs = self.retrieval_labels[view_id]

                        # b_prototype_vectors = prototype_vectors
                        # train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                        # train_labs = self.train_labels[view_id]
                        if loss_test < best_loss:
                            test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                            test_labs = self.test_labels[view_id]
                            retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                            retrieval_labs = self.retrieval_labels[view_id]

                            b_prototype_vectors = prototype_vectors
                            train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                            train_labs = self.train_labels[view_id]
                            best_loss = loss_test
                            # best_acc = acc
                            best_model_wts_pnet = copy.deepcopy(p_net.state_dict())
                            best_model_wts_pemb = p_net.get_pemb_weight()
                            best_epoch = epoch + 1
                                
                            view_labs = self.to_one_hot(self.test_labels[view_id])
                            view_data = self.to_one_hot(self.test_data[view_id])

                        #     data_fea = datanet(view_data, preprocess_flag=preprocess_flag)
                        #     pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                        #     running_corrects = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(view_labs, dim=1))
                        print(('ViewID: %d, Epoch %d/%d, l_loss: %.4f,  d_loss: %.4f, p_loss: %.4f, loss_test: %.4f') % (view_id, epoch + 1, self.epochs,loss1, loss2, loss3, loss_test))
            print('best_epoch:', best_epoch)
            torch.save(best_model_wts_pemb, 'models/' + self.datasets + '_' + str(0) + '_pemb.pth')
            torch.save(best_model_wts_pnet, 'models/' + self.datasets + '_' + str(view_id) + '_.pth')
            sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '.mat', {'test_fea':test_fea.sign().cpu().detach().numpy(),
                                                                                    'test_lab':test_labs,
                                                                                    'retrieval_fea':np.sign(retrieval_fea),
                                                                                    'retrieval_lab':retrieval_labs})
        # sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_valid.mat', {'valid_fea':valid_fea.sign().cpu().detach().numpy(),
        #                                                                         'valid_lab':valid_labs})
            sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '_train.mat', {'train_fea':train_fea,
                                                                                'train_lab':train_labs,
                                                                                })
            
            sio.savemat('Prototype/' + self.datasets + '_prototype_'  + '.mat', {'prototype': b_prototype_vectors.detach().cpu().numpy()})
        else:
            from model import Prototype_Net
            p_net = Prototype_Net(input_dim=self.input_shape[view_id],num_class=self.num_classes, 
                                  num_prototypes_per_class=self.num_prototypes_per_class, out_dim=self.bits).cuda()
            prototype_vectors_anchor = sio.loadmat('Prototype/' + self.datasets + '_prototype_' + '.mat')['prototype']
            prototype_vectors_anchor = torch.tensor(prototype_vectors_anchor, requires_grad=False).cuda()
            p_emb_inputs = self.init_pemb_input()

            get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
            params_pnet = get_grad_params(p_net)
            optimizer_pnet = optim.Adam(params_pnet, self.lr[view_id], [0.5, 0.999])
            best_acc = 0
            best_loss = 1e9
            best_epoch = 0
            for epoch in range(self.epochs):
                p_net.train()
                batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

                rand_didx = np.arange(self.train_data[view_id].shape[0])
                np.random.shuffle(rand_didx)

                for batch_idx in range(batch_nums):

                    didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                    view_labs = self.to_one_hot(self.train_labels[view_id][didx])
                    view_data = self.to_one_hot(self.train_data[view_id][didx])
                    optimizer_pnet.zero_grad()
                    if self.is_fine_tuning:
                        data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                        loss1 = klm_loss(sims, view_labs, tau=self.tau)
                        # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                        loss2 = dhl_loss(data_fea, view_labs, self.tau)
                        loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                        # loss3 = klr_orthogonal_loss(prototype_vectors, self.num_classes, tau=self.tau)
                        loss4 = klt_loss(prototype_vectors, prototype_vectors_anchor, self.bound, self.tau)
                        loss = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3 + self.beta2 * loss4
                    else:
                        data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs, prototype_vectors_anchor)
                        loss1 = klm_loss(sims, view_labs, tau=self.tau)
                        # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                        loss2 = dhl_loss(data_fea, view_labs, self.tau)
                        loss = self.eta * loss1 + self.alpha * loss2
                    loss.backward()
                    optimizer_pnet.step()

                    if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                        p_net.eval()
                        view_labs = self.to_one_hot(self.test_labels[view_id])
                        view_data = self.to_one_hot(self.test_data[view_id])
                        data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                        if self.is_fine_tuning:
                            loss1 = klm_loss(sims, view_labs, tau=self.tau)
                            # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                            loss2 = dhl_loss(data_fea, view_labs, self.tau)
                            loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                            # loss3 = klr_orthogonal_loss(prototype_vectors, self.num_classes, tau=self.tau)
                            loss4 = klt_loss(prototype_vectors, prototype_vectors_anchor, self.bound, self.tau)
                            loss_test = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3 + self.beta2 * loss4
                        else:
                            loss1 = klm_loss(sims, view_labs, tau=self.tau)
                            # loss1 = klm_loss_all(sims, view_labs, tau=self.tau)
                            loss2 = dhl_loss(data_fea, view_labs, self.tau)
                            loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                            loss_test = self.eta * loss1 + self.alpha * loss2
                        num_val = view_data.shape[0]
                        # acc = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(view_labs, dim=1)) / num_val

                        # test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                        # test_labs = self.test_labels[view_id]
                        # retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                        # retrieval_labs = self.retrieval_labels[view_id]

                        # b_prototype_vectors = prototype_vectors
                        # train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                        # train_labs = self.train_labels[view_id]
                        if loss_test < best_loss:
                            test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                            test_labs = self.test_labels[view_id]
                            retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                            retrieval_labs = self.retrieval_labels[view_id]

                            b_prototype_vectors = prototype_vectors
                            train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                            train_labs = self.train_labels[view_id]
                            best_loss = loss_test
                            # best_acc = acc
                            best_model_wts_pnet = copy.deepcopy(p_net.state_dict())
                            best_epoch = epoch + 1
                                
                            view_labs = self.to_one_hot(self.test_labels[view_id])
                            view_data = self.to_one_hot(self.test_data[view_id])

                        #     data_fea = datanet(view_data, preprocess_flag=preprocess_flag)
                        #     pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                        #     running_corrects = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(view_labs, dim=1))
                        print(('ViewID: %d, Epoch %d/%d, l_loss: %.4f,  d_loss: %.4f, p_loss: %.4f, loss_test: %.4f') % (view_id, epoch + 1, self.epochs,loss1, loss2, loss3, loss_test))
            print('best_epoch:', best_epoch)
            # torch.save(best_model_wts_datanet_1, 'models/' + self.datasets + '_' + str(view_id) + '_1_datanet.pth')
            torch.save(best_model_wts_pnet, 'models/' + self.datasets + '_' + str(view_id) + '_.pth')
            sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '.mat', {'test_fea':test_fea.sign().cpu().detach().numpy(),
                                                                                    'test_lab':test_labs,
                                                                                    'retrieval_fea':np.sign(retrieval_fea),
                                                                                    'retrieval_lab':retrieval_labs})
        # sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_valid.mat', {'valid_fea':valid_fea.sign().cpu().detach().numpy(),
        #                                                                         'valid_lab':valid_labs})
            sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '_train.mat', {'train_fea':train_fea,
                                                                                          'train_lab':train_labs})
    def evaluate_priors(self, view_id):
        from to_seed import to_seed
        to_seed(seed=self.seed)
        import os
        import torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        from model import Prototype_Net
        p_net = Prototype_Net(input_dim=self.input_shape[view_id],num_class=self.num_classes, 
                                  num_prototypes_per_class=self.num_prototypes_per_class, out_dim=self.bits).cuda()
        p_emb_inputs = self.init_pemb_input()
        get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
        params_pnet = get_grad_params(p_net)
        optimizer_pnet = optim.Adam(params_pnet, self.lr[view_id], [0.5, 0.999])
        best_acc = 0
        best_loss = 1e9
        best_epoch = 0
        best_score = 0
        for epoch in range(self.epochs):
            p_net.train()
            batch_nums = int(self.train_labels[view_id].shape[0] / float(self.batch_size))

            rand_didx = np.arange(self.train_data[view_id].shape[0])
            np.random.shuffle(rand_didx)

            for batch_idx in range(batch_nums):

                didx = rand_didx[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                view_labs = self.to_one_hot(self.train_labels[view_id][didx])
                view_data = self.to_one_hot(self.train_data[view_id][didx])
                optimizer_pnet.zero_grad()
                data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                loss1 = klm_loss(sims, view_labs, tau=self.tau)
                loss2 = dhl_loss(data_fea, view_labs, self.tau)
                loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                loss = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3

                loss.backward()
                optimizer_pnet.step()

                if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_nums - 1):
                    p_net.eval()
                    view_labs = self.to_one_hot(self.test_labels[view_id])
                    view_data = self.to_one_hot(self.test_data[view_id])
                    data_fea, prototype_vectors, sims = p_net(view_data, p_emb_inputs)
                    loss1 = klm_loss(sims, view_labs, tau=self.tau)
                    loss2 = dhl_loss(data_fea, view_labs, self.tau)
                    loss3 = klr_loss(prototype_vectors, self.num_classes, tau=self.tau)
                    loss_test = self.eta * loss1 + self.alpha * loss2 + self.beta1 * loss3


                    # test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                    # test_labs = self.test_labels[view_id]
                    # retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                    # retrieval_labs = self.retrieval_labels[view_id]

                    # b_prototype_vectors = prototype_vectors
                    # train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                    # train_labs = self.train_labels[view_id]
                    if loss_test < best_loss:
                        test_fea, prototype_vectors, _ = p_net(self.to_one_hot(self.test_data[view_id]), p_emb_inputs)
                        test_labs = self.test_labels[view_id]
                        retrieval_fea = utils.predict(p_net, self.to_one_hot(self.retrieval_data[view_id]), p_emb_inputs)
                        retrieval_labs = self.retrieval_labels[view_id]

                        b_prototype_vectors = prototype_vectors
                        train_fea = utils.predict(p_net, self.to_one_hot(self.train_data[view_id]), p_emb_inputs)
                        train_labs = self.train_labels[view_id]
                        cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6)
                        sims = cos(torch.tensor(train_fea, requires_grad=False).cuda() , b_prototype_vectors)   
                        score1 = klm_score(sims, self.to_one_hot(train_labs))
                        score2 = klr_score(b_prototype_vectors, self.num_classes, tau=self.tau)
                        prior_score = (score1 + score2) / 2


                        best_score = max(prior_score, best_score)
                        

                        best_loss = loss_test
                        # best_acc = acc
                        best_model_wts_pnet = copy.deepcopy(p_net.state_dict())
                        best_model_wts_pemb = p_net.get_pemb_weight()
                        best_epoch = epoch + 1
                                
                        view_labs = self.to_one_hot(self.test_labels[view_id])
                        view_data = self.to_one_hot(self.test_data[view_id])

                        # data_fea = datanet(view_data, preprocess_flag=preprocess_flag)
                        # pred = data_fea.view([data_fea.shape[0], -1]).mm(W)
                        # running_corrects = torch.sum(torch.argmax(pred, dim=1) == torch.argmax(view_labs, dim=1))
                    print(('ViewID: %d, Epoch %d/%d, l_loss: %.4f,  d_loss: %.4f, p_loss: %.4f, loss_test: %.4f') % (view_id, epoch + 1, self.epochs,loss1, loss2, loss3, loss_test))
        print('best_epoch:', best_epoch, 'best_score:', best_score)
        # torch.save(best_model_wts_pemb, 'models/' + self.datasets + '_' + str(0) + '_pemb.pth')
        # sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '.mat', {'test_fea':test_fea.sign().cpu().detach().numpy(),
        #                                                                             'test_lab':test_labs,
        #                                                                             'retrieval_fea':np.sign(retrieval_fea),
        #                                                                             'retrieval_lab':retrieval_labs})
        # sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_valid.mat', {'valid_fea':valid_fea.sign().cpu().detach().numpy(),
        #                                                                         'valid_lab':valid_labs})
        sio.savemat('features/' + self.datasets + '_' + str(view_id) + '_' + str(self.bits) + '_train.mat', {'train_fea':train_fea,
                                                                                'train_lab':train_labs,
                                                                                })
            
        sio.savemat('Prototype/' + self.datasets + '_prototype_'  + '.mat', {'prototype': b_prototype_vectors.detach().cpu().numpy()})