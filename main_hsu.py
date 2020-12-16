import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger, logfile
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from typing import Sequence
import torch
import torch.nn as nn
from collections import deque
from tqdm import tqdm
from typing import Optional, Mapping

from deepxml.evaluation import get_p_5, get_n_5
from deepxml.optimizers import DenseSparseAdam
from deepxml.xmlcnn_hsu import CorNetXMLCNN

import parameter

pa = parameter.Parameter()
import warnings

warnings.filterwarnings("ignore", category=Warning)
logfile(pa.main_log)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def output_res(output_path, name, scores, labels):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, F'{name}-scores'), scores)
    np.save(os.path.join(output_path, F'{name}-labels'), labels)


TDataX = Sequence[Sequence]
TDataY = Optional[csr_matrix]


class MultiLabelDataset(Dataset):
    def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True):
        self.data_x, self.data_y, self.training = data_x, data_y, training

    def __getitem__(self, item):
        data_x = self.data_x[item]
        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.data_x)


class Model(object):
    def __init__(self, network, model_path, gradient_clip_value=5.0, device_ids=None, **kwargs):
        print("__init__")
        # print(network, model_path, gradient_clip_value, device_ids)
        self.model = nn.DataParallel(network(**kwargs).to(device), device_ids=device_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model_path = model_path
        os.makedirs(os.path.split(self.model_path)[0], exist_ok=True)
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        print("train_step")
        # train_x: 40 x 500, train_y: 40 x 3801
        self.optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss關於weight的導數變成0.

        self.model.train()  # 從評估模式轉為訓練模式
        print("train_x:",train_x.shape)
        scores = self.model(train_x)  # 40 x 3801
        loss = self.loss_fn(scores, train_y)
        # print("loss = ", loss)            tensor(0.6634, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
        loss.backward()
        # print("loss.backward() = ", loss) tensor(0.6634, grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: torch.Tensor, k: int):
        print("predict_step")

        self.model.eval()
        with torch.no_grad():
            scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu()

    def get_optimizer(self, **kwargs):
        print("get_optimizer")
        self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=1, step=10, k=5, early=50, verbose=True, **kwargs):
        print("train")  # {'batch_size': 40, 'nb_epoch': 30, 'swa_warmup': 10}
        nb_epoch = 5
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        print_loss = 0.0  #
        print("len(train_loader) = %d, len(valid_loader) = %d" % (len(train_loader), len(valid_loader)))
        for epoch_idx in range(nb_epoch):
            for i, (train_x, train_y) in enumerate(train_loader, 1):
                # train_x: 40 x 500, train_y: 40 x 3801
                global_step += 1
                loss = self.train_step(train_x, train_y.to(device))
                # print("return loss =", loss)
                print_loss += loss  #
                if global_step % step == 0:
                    print("detail: epoch_idx = %s, i = %s" % (epoch_idx, i))
                    labels = []
                    valid_loss = 0.0
                    self.model.eval()  # 從訓練模式轉為評估模式
                    with torch.no_grad():  # 不需要計算導數
                        for (valid_x, valid_y) in valid_loader:
                            logits = self.model(valid_x)
                            valid_loss += self.loss_fn(logits, valid_y.to(device)).item()
                            scores, tmp = torch.topk(logits, k)  # (40, 5)
                            labels.append(tmp.cpu())
                    valid_loss /= len(valid_loader)
                    labels = np.concatenate(labels)
                    # print(labels[0])  # (200, 5)
                    targets = valid_loader.dataset.data_y
                    p5, n5 = get_p_5(labels, targets), get_n_5(labels, targets)  # ndcg
                    print(p5, n5)
                    if n5 > best_n5:
                        self.save_model()  # epoch_idx > 1 * swa_warmup)
                        best_n5, e = n5, 0
                    else:
                        e += 1
                        if early is not None and e > early:
                            return
                    if True:
                        log_msg = '%d %d train loss: %.7f valid loss: %.7f P@5: %.5f N@5: %.5f early stop: %d' % \
                                  (epoch_idx, i * train_loader.batch_size, print_loss / step,
                                   valid_loss, round(p5, 5), round(n5, 5), e)
                        logger.info(log_msg)
                        print_loss = 0.0

    def predict(self, data_loader: DataLoader, k=100, desc='Predict'):
        print("k= ", k)
        self.load_model()
        scores_list, labels_list = zip(*(self.predict_step(data_x, k)
                                         for data_x in tqdm(data_loader, desc=desc, leave=False)))
        return np.concatenate(scores_list), np.concatenate(labels_list)

    def save_model(self):
        for trial in range(5):
            try:
                torch.save(self.model.module.state_dict(), self.model_path)
                break
            except:
                print('saving failed')

    def load_model(self):
        print("load: ", self.model_path)
        self.model.module.load_state_dict(torch.load(self.model_path))

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))
            if total_norm > max_norm * self.gradient_clip_value:
                logger.warn(F'Clipping gradients with total norm {round(total_norm, 5)} '
                            F'and max norm {round(max_norm, 5)}')


# @click.command()
# @click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
# @click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
# @click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
def main(data_cnf, model_cnf, mode=None):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))  # 讀取yaml

    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    print("使用的model: %s, 資料集: %s" % (model_name, data_name))
    model_path = os.path.join(pa.models, F'{model_name}-{data_name}')
    print("存放model的path :", model_path)
    emb_init = np.load(pa.emb)
    print("emb_init.shape = ", emb_init.shape)  # (166402, 300)
    logger.info(F'Model Name: {model_name}')

    network = CorNetXMLCNN

    if mode is not None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x = np.load(pa.train_texts, allow_pickle=True)  # (15449, 500)
        train_labels = np.load(pa.train_labels, allow_pickle=True)  # (15449,)
        # train_x, train_labels = train_x[:480], train_labels[:480]

        random_state = 1240
        # 分割 valid_x
        train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels, test_size=200,
                                                                        random_state=random_state)  # 1240
        mlb = get_mlb(pa.labels_binarizer, np.hstack((train_labels, valid_labels)))
        # <class 'sklearn.preprocessing._label.MultiLabelBinarizer'>
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)  # (15249, 3801) (200, 3801)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')  # 3801
        logger.info(F'Size of Training Set: {len(train_x)}')  # 15249
        logger.info(F'Size of Validation Set: {len(valid_x)}')  # 200

        logger.info('Training')
        # num_workers=4 num_workers=4 num_workers=4 num_workers=4 num_workers=4
        train_loader = DataLoader(MultiLabelDataset(train_x, train_y),  # train_x: 15249 x 500, train_y: 15249 x 3801
                                  batch_size=40, shuffle=True)  # batch_size = 40
        valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=True),
                                  # valid_x: 200 x 500, valid_y: 200 x 3801
                                  batch_size=40)  # batch_size = 40

        model = Model(network=network, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                      **data_cnf['model'], **model_cnf['model'])

        # print(model_cnf['train']) {'batch_size': 40, 'nb_epoch': 30, 'swa_warmup': 10}
        model.train(train_loader, valid_loader, **model_cnf['train'])
        logger.info('Finish Training')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        mlb = get_mlb(pa.labels_binarizer)
        labels_num = len(mlb.classes_)  # 3801
        test_x = np.load(pa.test_texts, allow_pickle=True)
        # test_labels = np.load(pa.test_labels, allow_pickle=True)
        # test_y = mlb.transform(test_labels)
        # test_x = test_x[:400]
        logger.info(F'Size of Test Set: {len(test_x)}')
        logger.info('Predicting')
        print("model_cnf['predict']['batch_size'] = ", model_cnf['predict']['batch_size'])
        test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'])
        if model is None:
            model = Model(network=network, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                          **data_cnf['model'], **model_cnf['model'])
        # print(model_cnf['predict'])  # {'batch_size': 40}
        scores, labels = model.predict(test_loader, k=1)
        print(scores.shape, labels.shape)
        logger.info('Finish Predicting')
        labels = mlb.classes_[labels]
        output_res(pa.results, F'{model_name}-{data_name}', scores, labels)


if __name__ == '__main__':
    data_cnf = pa.data_cnf
    # model_cnf = pa.CorNet_model_cnf
    model_cnf = pa.model_cnf
    mode = "train"
    main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)
