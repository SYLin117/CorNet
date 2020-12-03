import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model, GPipeModel
from deepxml.attentionxml import AttentionXML, CorNetAttentionXML
from deepxml.meshprobenet import MeSHProbeNet, CorNetMeSHProbeNet
from deepxml.bertxml import BertXML, CorNetBertXML
from deepxml.xmlcnn import XMLCNN, CorNetXMLCNN

import parameter
pa = parameter.Parameter()

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


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
def main(data_cnf, model_cnf, mode):
    data_cnf = pa.data_cnf
    model_cnf = pa.model_cnf

    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    print(data_cnf['labels_binarizer'])
    print("~~~~~~~~~~~~~~~")
    print(data_cnf['embedding'])
    print(data_cnf['valid'])
    print(model_cnf)
    print(data_cnf['model'])
    print(model_cnf['model'])

    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    print(model, model_name, data_name)  # None XMLCNN EUR-Lex
    model_path = os.path.join(pa.models, F'{model_name}-{data_name}')
    print(model_path)
    emb_init = get_word_emb(pa.emb)
    print(emb_init.shape)  # (166402, 300)
    logger.info(F'Model Name: {model_name}')

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(pa.train_texts, pa.train_labels)
        print(train_x.shape, train_labels.shape)  # (15449, 500) (15449,)
        if 'size' in data_cnf['valid']:
            random_state = data_cnf['valid'].get('random_state', 1240)
            print(random_state)  # 1240
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(pa.labels_binarizer, np.hstack((train_labels, valid_labels)))
        print(type(mlb))  # <class 'sklearn.preprocessing._label.MultiLabelBinarizer'>
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')  # 3801
        logger.info(F'Size of Training Set: {len(train_x)}')  # 15249
        logger.info(F'Size of Validation Set: {len(valid_x)}')  # 200

        logger.info('Training')
        print("Training")
        print(model_cnf['train']['batch_size'])  # 40
        train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                  model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
        print(model_cnf['valid']['batch_size']) # 40
        valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=True),
                                  model_cnf['valid']['batch_size'], num_workers=4)
        if 'gpipe' not in model_cnf:
            print("not in")
            model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                          **data_cnf['model'], **model_cnf['model'])
        else:
            print("in")
            model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                               **data_cnf['model'], **model_cnf['model'])
        model.train(train_loader, valid_loader, **model_cnf['train'])
        logger.info('Finish Training')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        logger.info('Predicting')
        test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'], num_workers=4)
        if 'gpipe' not in model_cnf:
            if model is None:
                model = Model(network=model_dict[model_name], labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                              **data_cnf['model'], **model_cnf['model'])
        else:
            if model is None:
                model = GPipeModel(model_name, labels_num=labels_num, model_path=model_path, emb_init=emb_init, 
                                   **data_cnf['model'], **model_cnf['model'])
        scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
        logger.info('Finish Predicting')
        labels = mlb.classes_[labels]
        output_res(data_cnf['output']['res'], F'{model_name}-{data_name}', scores, labels)

tl = "/Users/mm/Documents/Course_Information/Data_Mining/EUR-Lex/train_labels.txt"

if __name__ == '__main__':
    main()
