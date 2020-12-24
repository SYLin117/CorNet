import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from logzero import logger, logfile

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model, GPipeModel
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.meshprobenet import MeSHProbeNet, CorNetMeSHProbeNet
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

import parameter

model_dict = {
    'AttentionXML': AttentionXML,
    'CorNetAttentionXML': CorNetAttentionXML,
    'MeSHProbeNet': MeSHProbeNet,
    'CorNetMeSHProbeNet': CorNetMeSHProbeNet,
    'BertXML': BertXML,
    'CorNetBertXML': CorNetBertXML,
    'XMLCNN': XMLCNN,
    'CorNetXMLCNN': CorNetXMLCNN
}


# @click.command()
# @click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
# @click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
# @click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
def main(data_cnf, model_cnf, mode):
    model_name = os.path.split(model_cnf)[1].split(".")[0]
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    # 設定log檔案位置
    logfile(
        "./logs/logfile_{0}_cornet_{1}_cornet_dim_{2}.log".format(model_name, model_cnf['model']['n_cornet_blocks'],
                                                                  model_cnf['model']['cornet_dim']))

    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'],
                              F'{model_name}-{data_name}-{model_cnf["model"]["n_cornet_blocks"]}-{model_cnf["model"]["cornet_dim"]}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')
    # summary(model_dict[model_name])
    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
        if 'size' in data_cnf['valid']:
            random_state = data_cnf['valid'].get('random_state', 1240)
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_x)}')
        logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                  model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
        valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=True),
                                  model_cnf['valid']['batch_size'], num_workers=4)

        if 'gpipe' not in model_cnf:
            model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path,
                          emb_init=emb_init,
                          **data_cnf['model'], **model_cnf['model'])
        else:
            model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                               **data_cnf['model'], **model_cnf['model'])
        loss, p1, p5 = model.train(train_loader, valid_loader, **model_cnf['train'])
        np.save(model_cnf['np_loss'] + "{0}_cornet_{1}_cornet_dim_{2}.npy".format(model_name,
                                                                                  model_cnf['model'][
                                                                                      'n_cornet_blocks'],
                                                                                  model_cnf['model'][
                                                                                      'cornet_dim']),
                loss)
        np.save(model_cnf['np_p1'] + "{0}_cornet_{1}_cornet_dim_{2}.npy".format(model_name,
                                                                                model_cnf['model'][
                                                                                    'n_cornet_blocks'],
                                                                                model_cnf['model'][
                                                                                    'cornet_dim']),
                p1)
        np.save(model_cnf['np_p5'] + "{0}_cornet_{1}_cornet_dim_{2}.npy".format(model_name,
                                                                                model_cnf['model'][
                                                                                    'n_cornet_blocks'],
                                                                                model_cnf['model'][
                                                                                    'cornet_dim']),
                p5)
        logger.info('Finish Training')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        logger.info('model path: ', model_path)
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        logger.info('Predicting')
        test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=4)
        if 'gpipe' not in model_cnf:
            if model is None:
                model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path,
                              emb_init=emb_init,
                              **data_cnf['model'], **model_cnf['model'])
        else:
            if model is None:
                model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                                   **data_cnf['model'], **model_cnf['model'])
        scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
        logger.info('Finish Predicting')
        labels = mlb.classes_[labels]
        output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', scores, labels)


if __name__ == '__main__':
    # print("torch cuda is available: ", torch.cuda.is_available())
    PROJECT_CONF = "E:/PycharmProject/CorNet/configure/"
    param = parameter.Parameter()
    data_cnf = param.data_cnf
    model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex.yaml"
    mode = "train"
    main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)

    # model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex2.yaml"
    # main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)

    # model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex3.yaml"
    # main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)

    # model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex4.yaml"
    # main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)

    # model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex5.yaml"
    # main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)

    # model_cnf = PROJECT_CONF + "models/CorNetXMLCNN-EUR-Lex6.yaml"
    # main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)
