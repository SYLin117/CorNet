import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from typing import Optional, Mapping

from deepxml.evaluation import get_p_5, get_n_5, get_p_1, get_n_1
from deepxml.optimizers import DenseSparseAdam
from deepxml.gpipe_modules import gpipe_encoder, gpipe_decoder

from torchgpipe import GPipe
from torchsummary import summary

from torchviz import make_dot, make_dot_from_trace
import time
class Model(object):

    def __init__(self, network, model_path, gradient_clip_value=5.0, device_ids=None, **kwargs):
        """

        :param network: 網路model
        :param model_path: train好的儲存位置
        :param gradient_clip_value:
        :param device_ids:
        :param kwargs:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = network(**kwargs).to(self.device)

        self.model = nn.DataParallel(model, device_ids=device_ids)
        # 顯示model架構用
        # model: XMLCNN(
        #     (emb): Embedding(166402, 300, padding_idx=0, sparse=True)
        # (conv1): Conv2d(1, 128, kernel_size=(2, 300), stride=(1, 1), padding=(1, 0))
        # (conv2): Conv2d(1, 128, kernel_size=(4, 300), stride=(1, 1), padding=(3, 0))
        # (conv3): Conv2d(1, 128, kernel_size=(8, 300), stride=(1, 1), padding=(7, 0))
        # (pool): AdaptiveMaxPool1d(output_size=8)
        # (bottleneck): Linear(in_features=3072, out_features=512, bias=True)
        # (dropout): Dropout(p=0.5, inplace=False)
        # (fc1): Linear(in_features=512, out_features=3801, bias=True)
        # )
        # print("model:", model)
        # summary(model, input_size=(1, 40, 500))

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path, self.state = model_path, {}
        os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        # 原始寫法
        # scores = self.model(train_x)
        # 原本會出現error: RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.cuda.DoubleTensor instead (while checking arguments for embedding)
        scores = self.model(train_x.long())

        #把model劃出來
        dot = make_dot(scores, params=dict(list(self.model.named_parameters()) + [('train_x', train_x)]))
        dot.render('./pytorch_cornet')
        #time.sleep(10000)

        loss = self.loss_fn(scores, train_y)
        loss.backward()
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: torch.Tensor, k: int):
        self.model.eval()
        with torch.no_grad():
            scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu()

    def get_optimizer(self, **kwargs):
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=100, step=100, k=5, early=100, verbose=True, swa_warmup=None, **kwargs):
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        print_loss = 0.0  #
        epoch_loss = np.zeros((1, nb_epoch), dtype=np.float)
        np_p1 = np.zeros((1, nb_epoch), dtype=np.float)
        np_p5 = np.zeros((1, nb_epoch), dtype=np.float)
        for epoch_idx in range(nb_epoch):
            print_epoch_loss = 0.0
            if epoch_idx == swa_warmup:
                self.swa_init()
            p1_tmp = 0.0
            p5_tmp = 0.0
            for i, (train_x, train_y) in enumerate(train_loader, 1):
                global_step += 1
                loss = self.train_step(train_x, train_y.cuda())
                print_loss += loss
                print_epoch_loss += loss
                if global_step % step == 0:
                    self.swa_step()
                    self.swap_swa_params()
                    ##
                    labels = []
                    valid_loss = 0.0
                    self.model.eval()
                    with torch.no_grad():
                        for (valid_x, valid_y) in valid_loader:
                            # 原始寫法
                            # logits = self.model(valid_x)
                            logits = self.model(valid_x.long())
                            valid_loss += self.loss_fn(logits, valid_y.cuda()).item()
                            scores, tmp = torch.topk(logits, k)
                            labels.append(tmp.cpu())
                    valid_loss /= len(valid_loader)
                    labels = np.concatenate(labels)
                    ##
                    #                    labels = np.concatenate([self.predict_step(valid_x, k)[1] for valid_x in valid_loader])
                    targets = valid_loader.dataset.data_y
                    p5, n5 = get_p_5(labels, targets), get_n_5(labels, targets)
                    p1, n1 = get_p_1(labels, targets), get_n_1(labels, targets)
                    p5_tmp = p5
                    p1_tmp = p1
                    if n5 > best_n5:
                        self.save_model(True)  # epoch_idx > 1 * swa_warmup)
                        best_n5, e = n5, 0
                    else:
                        e += 1
                        if early is not None and e > early:
                            return
                    self.swap_swa_params()
                    if verbose:
                        log_msg = '%d %d train loss: %.7f valid loss: %.7f P@5: %.5f N@5: %.5f P@1: %.5f N@1: %.5f early stop: %d' % \
                                  (epoch_idx, i * train_loader.batch_size, print_loss / step, valid_loss, round(p5, 5),
                                   round(n5, 5), round(p1, 5), round(n1, 5), e)
                        logger.info(log_msg)
                        print_loss = 0.0

            epoch_loss[0, epoch_idx] = print_epoch_loss / 15249
            print_epoch_loss = 0.0
            np_p1[0, epoch_idx] = p1_tmp
            np_p5[0, epoch_idx] = p5_tmp
        return epoch_loss, np_p1, np_p5

    def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
        self.load_model()
        scores_list, labels_list = zip(*(self.predict_step(data_x, k)
                                         for data_x in tqdm(data_loader, desc=desc, leave=False)))
        return np.concatenate(scores_list), np.concatenate(labels_list)

    def save_model(self, last_epoch):
        if not last_epoch: return
        for trial in range(5):
            try:
                torch.save(self.model.module.state_dict(), self.model_path)
                break
            except:
                print('saving failed')

    def load_model(self):
        self.model.module.load_state_dict(torch.load(self.model_path))

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
                            F'and max norm {round(max_norm, 5)}')

    def swa_init(self):
        if 'swa' not in self.state:
            logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.cpu().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n].cuda(), p.data.cpu()

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']


class GPipeModel(object):
    def __init__(self, model_name, model_path, gradient_clip_value=5.0, device_ids=None, **kwargs):
        gpipe_model = nn.Sequential(gpipe_encoder(model_name, **kwargs), gpipe_decoder(model_name, **kwargs))
        self.model = GPipe(gpipe_model, balance=[1, 1], chunks=2)
        self.in_device = self.model.devices[0]
        self.out_device = self.model.devices[-1]
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path, self.state = model_path, {}
        os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(train_x)
        loss = self.loss_fn(scores, train_y)
        loss.backward()
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: torch.Tensor, k: int):
        self.model.eval()
        with torch.no_grad():
            scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu()

    def get_optimizer(self, **kwargs):
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=100, step=100, k=5, early=100, verbose=True, swa_warmup=None, **kwargs):
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        print_loss = 0.0  #
        for epoch_idx in range(nb_epoch):
            if epoch_idx == swa_warmup:
                self.swa_init()
            for i, (train_x, train_y) in enumerate(train_loader, 1):
                global_step += 1
                loss = self.train_step(train_x.to(self.in_device, non_blocking=True),
                                       train_y.to(self.out_device, non_blocking=True))
                print_loss += loss  #
                if global_step % step == 0:
                    self.swa_step()
                    self.swap_swa_params()
                    ##
                    labels = []
                    valid_loss = 0.0
                    self.model.eval()
                    with torch.no_grad():
                        for (valid_x, valid_y) in valid_loader:
                            logits = self.model(valid_x.to(self.in_device, non_blocking=True))
                            valid_loss += self.loss_fn(logits, valid_y.to(self.out_device, non_blocking=True)).item()
                            scores, tmp = torch.topk(logits, k)
                            labels.append(tmp.cpu())
                    valid_loss /= len(valid_loader)
                    labels = np.concatenate(labels)
                    ##
                    #                    labels = np.concatenate([self.predict_step(valid_x, k)[1] for valid_x in valid_loader])
                    targets = valid_loader.dataset.data_y
                    p5, n5 = get_p_5(labels, targets), get_n_5(labels, targets)
                    if n5 > best_n5:
                        self.save_model(epoch_idx > 3 * swa_warmup)
                        best_n5, e = n5, 0
                    else:
                        e += 1
                        if early is not None and e > early:
                            return
                    self.swap_swa_params()
                    if verbose:
                        log_msg = '%d %d train loss: %.7f valid loss: %.7f P@5: %.5f N@5: %.5f early stop: %d' % \
                                  (epoch_idx, i * train_loader.batch_size, print_loss / step, valid_loss, round(p5, 5),
                                   round(n5, 5), e)
                        logger.info(log_msg)
                        print_loss = 0.0

    def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
        self.load_model()
        scores_list, labels_list = zip(*(self.predict_step(data_x.to(self.in_device, non_blocking=True), k)
                                         for data_x in tqdm(data_loader, desc=desc, leave=False)))
        return np.concatenate(scores_list), np.concatenate(labels_list)

    def save_model(self, last_epoch):
        if not last_epoch: return
        for trial in range(5):
            try:
                torch.save(self.model.state_dict(), self.model_path)
                break
            except:
                print('saving failed')

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
                            F'and max norm {round(max_norm, 5)}')

    def swa_init(self):
        if 'swa' not in self.state:
            logger.info('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.cpu().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                gpu_id = p.get_device()
                p.data, swa_state[n] = swa_state[n], p.data.cpu()  #
                p.data = p.data.cuda(gpu_id)

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']
