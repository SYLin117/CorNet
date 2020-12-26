import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Embedding, Dropout, GlobalMaxPooling1D, Input, Convolution1D, Convolution2D, \
    BatchNormalization, Activation, MaxPooling2D, GlobalMaxPool2D, Lambda, concatenate
import tensorflow as tf

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res

from logzero import logger, logfile
import os
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split

import math


def batch_generator(X_data, y_data, batch_size):
    """
    为什么我们需要steps_per_epoch？
    请记住，Keras数据生成器意味着无限循环，它永远不会返回或退出。
    :param X_data:
    :param y_data:
    :param batch_size:
    :return:
    """
    # 總資料比數
    samples_per_epoch = X_data.shape[0]
    # 在batch_size的大小下有多少的batch
    number_of_batches = samples_per_epoch / batch_size
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X_data[index_batch, :]
        y_batch = y_data[index_batch].todense()
        if (len(index_batch) < 40):
            counter = 0
            continue
        counter += 1
        yield X_batch, np.array(y_batch)
        if (counter > number_of_batches):
            counter = 0


def adapmaxpooling(x, outsize):
    x_shape = keras.backend.int_shape(x)
    batchsize1, dim1, dim2, channels1 = x_shape
    stride = np.floor(dim1 / outsize).astype(np.int32)
    kernels = dim1 - (outsize - 1) * stride
    adpooling = MaxPooling2D(pool_size=(kernels, kernels), strides=stride)(x)

    return adpooling


def squeeze_function(x, dim):
    return keras.backend.squeeze(x, axis=dim)


def reshape_tensor(x, shape):
    return keras.backend.reshape(x, shape)


def main(data_cnf, model_cnf, mode):
    model_name = os.path.split(model_cnf)[1].split(".")[0]
    # 設定log檔案位置
    logfile("./logs/logfile_" + model_name + ".log")
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    # model_path = model_cnf['path'] + "/" + model_cnf['name'] + '.h'
    model_path = r'E:\\PycharmProject\\CorNet\\keras_xmlcnn.h5'
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

        # train_x.reshape((15249, 1, 500, 1))
        # valid_x.reshape((200, 1, 500, 1))
        # train_y = LabelBinarizer(sparse_output=True).fit(labels).transform(Y)

        vocab_size = emb_init.shape[0]
        emb_size = emb_init.shape[1]

        # 可調參數
        data_num = len(train_x)
        ks = 3
        output_channel = model_cnf['model']['num_filters']
        dynamic_pool_length = model_cnf['model']['dynamic_pool_length']
        num_bottleneck_hidden = model_cnf['model']['bottleneck_dim']
        drop_out = model_cnf['model']['dropout']
        nb_epochs = model_cnf['train']['nb_epoch']
        batch_size = model_cnf['train']['batch_size']
        max_length = 500

        input_tensor = keras.Input(shape=(max_length,), name='input')
        emb_data = Embedding(input_dim=vocab_size,
                             output_dim=emb_size,
                             input_length=max_length,
                             weights=[emb_init],
                             trainable=False,
                             name='embedding1')(input_tensor)
        emb_data.trainable = False
        # emd_out_4d = keras.layers.core.RepeatVector(1)(emb_data)
        # unsqueeze_emb_data = tf.keras.layers.Reshape((1, 500, 300), input_shape=(500, 300))(emb_data)
        # emb_data = tf.expand_dims(emb_data, axis=1)
        # emb_data = Lambda(reshape_tensor, arguments={'shape': (1, max_length, 300)}, name='lambda1')(
        #     emb_data)

        conv1_output = Convolution1D(output_channel, 2, padding='same',
                                     kernel_initializer=keras.initializers.glorot_uniform(seed=None),
                                     activation='relu', name='conv1')(emb_data)
        # conv1_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, max_length, output_channel)},
        #                       name='conv1_lambda')(
        #     conv1_output)

        conv2_output = Convolution1D(output_channel, 4, padding='same',
                                     kernel_initializer=keras.initializers.glorot_uniform(seed=None),
                                     activation='relu', name='conv2')(emb_data)
        # conv2_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, max_length, output_channel)},
        #                       name='conv2_lambda')(
        #     conv2_output)

        conv3_output = Convolution1D(output_channel, 8, padding='same',
                                     kernel_initializer=keras.initializers.glorot_uniform(seed=None),
                                     activation='relu', name='conv3')(emb_data)
        # conv3_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, max_length, output_channel)},
        #                       name='conv3_lambda')(
        #     conv3_output)

        pool1 = GlobalMaxPooling1D(name='globalmaxpooling1')(conv1_output)
        pool2 = GlobalMaxPooling1D(name='globalmaxpooling2')(conv2_output)
        pool3 = GlobalMaxPooling1D(name='globalmaxpooling3')(conv3_output)
        output = concatenate([pool1, pool2, pool3], axis=-1)
        # output = Dense(num_bottleneck_hidden, activation='relu',name='bottleneck')(output)
        output = Dropout(0.5, name='dropout1')(output)
        output = Dense(3801, activation='softmax', name='dense_final',
                       kernel_initializer=keras.initializers.glorot_uniform(seed=None))(output)
        model = Model(input_tensor, output)
        model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(top_k=5)])
        history = model.fit_generator(steps_per_epoch=data_num / batch_size,
                                      generator=batch_generator(train_x, train_y, batch_size),
                                      nb_epoch=nb_epochs,
                                      validation_data=batch_generator(valid_x, valid_y, batch_size),
                                      nb_val_samples=valid_x.shape[0])
        model.save(model_path)
        if mode is None or mode == 'eval':
            # logger.info('Loading Training and Validation Set')
            # train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
            # if 'size' in data_cnf['valid']:
            #     random_state = data_cnf['valid'].get('random_state', 1240)
            #     train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
            #                                                                     test_size=data_cnf['valid']['size'],
            #                                                                     random_state=random_state)
            # else:
            #     valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
            # mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
            # train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
            # labels_num = len(mlb.classes_)
            # logger.info(F'Number of Labels: {labels_num}')
            # logger.info(F'Size of Training Set: {len(train_x)}')
            # logger.info(F'Size of Validation Set: {len(valid_x)}')

            logger.info('Loading Test Set')
        logger.info('model path: ', model_path)
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        model = keras.models.load_model(model_path)
        model.predict(test_x)

        if __name__ == '__main__':
            # print("torch cuda is available: ", torch.cuda.is_available())
            PROJECT_CONF = "E:/PycharmProject/CorNet/configure/"
        data_cnf = PROJECT_CONF + "datasets/EUR-Lex.yaml"
        model_cnf = PROJECT_CONF + "models/Keras-CorNetXMLCNN-EUR-Lex.yaml"
        mode = "train"
        main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)
