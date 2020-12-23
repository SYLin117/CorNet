from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Conv1D, Conv2D, Dropout, GlobalMaxPooling1D, Input
import tensorflow as tf

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res

from logzero import logger, logfile
import os
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split

import parameter


def main(data_cnf, model_cnf, mode):
    model_name = os.path.split(model_cnf)[1].split(".")[0]
    # 設定log檔案位置
    logfile("./logs/logfile_" + model_name + ".log")
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

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

        train_x.reshape((15249, 1, 500, 1))
        valid_x.reshape((200, 1, 500, 1))
        train_y = LabelBinarizer(sparse_output=True).fit(labels).transform(Y)

    vocab_size = emb_init.shape[0]
    emb_size = emb_init.shape[1]

    # 可調參數
    ks = 3
    output_channel = 128
    dynamic_pool_length = 8
    num_bottleneck_hidden = 512
    drop_out = 0.5
    epochs = 30
    batch_size = 40
    glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    glorot_normal_initializer = tf.compat.v1.keras.initializers.glorot_normal()
    model = Sequential()
    embedding_layer = Embedding(vocab_size,
                                emb_size,
                                weights=[emb_init],
                                input_length=500,
                                trainable=False)
    model.add(embedding_layer)
    model.add(Conv2D(output_channel, kernel_size=(2, emb_size), padding='same',
                     kernel_initializer=glorot_uniform_initializer, input_shape=(1, 500, 1)))
    model.add(Conv2D(output_channel, kernel_size=(4, emb_size), padding='same',
                     kernel_initializer=glorot_uniform_initializer))
    model.add(Conv2D(output_channel, kernel_size=(8, emb_size), padding='same',
                     kernel_initializer=glorot_uniform_initializer))
    model.add(GlobalMaxPooling1D(data_format=str(dynamic_pool_length)))
    model.add(Dense(num_bottleneck_hidden, input_shape=(ks * output_channel * dynamic_pool_length,), activation='relu',
                    initializer=glorot_uniform_initializer))
    model.add(Dropout(drop_out))
    model.add(Dense(labels_num, input_shape=(num_bottleneck_hidden,), activation='relu',
                    initializer=glorot_uniform_initializer))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(top_k=5)])
    print(model.summary())
    history = model.fit_generator(train_x, valid_x, epochs=epochs, batch_size=batch_size,
                        validation_data=(train_y, valid_y.toArray()))


if __name__ == '__main__':
    # print("torch cuda is available: ", torch.cuda.is_available())
    PROJECT_CONF = "E:/PycharmProject/CorNet/configure/"
    data_cnf = PROJECT_CONF + "datasets/EUR-Lex.yaml"
    model_cnf = PROJECT_CONF + "models/Keras-CorNetXMLCNN-EUR-Lex.yaml"
    mode = "train"
    main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)
