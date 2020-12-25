import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Embedding, Conv1D, Conv2D, Dropout, GlobalMaxPooling1D, Input, Convolution2D, \
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

        # train_x.reshape((15249, 1, 500, 1))
        # valid_x.reshape((200, 1, 500, 1))
        # train_y = LabelBinarizer(sparse_output=True).fit(labels).transform(Y)

    vocab_size = emb_init.shape[0]
    emb_size = emb_init.shape[1]

    # 可調參數
    data_num = len(train_x)
    ks = 3
    output_channel = 128
    dynamic_pool_length = 8
    num_bottleneck_hidden = 512
    drop_out = 0.5
    nb_epochs = 30
    batch_size = 40
    glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    glorot_normal_initializer = tf.compat.v1.keras.initializers.glorot_normal()

    input_tensor = keras.Input(batch_shape=(batch_size, 500), name='text')
    emb_data = Embedding(vocab_size,
                         emb_size,
                         weights=[emb_init],
                         trainable=False)(input_tensor)
    # emd_out_4d = keras.layers.core.RepeatVector(1)(emb_data)
    # unsqueeze_emb_data = tf.keras.layers.Reshape((1, 500, 300), input_shape=(500, 300))(emb_data)
    # emb_data = tf.expand_dims(emb_data, axis=1)
    emb_data = Lambda(reshape_tensor, arguments={'shape': (batch_size, 1, 500, 300)})(emb_data)

    conv1_output = Convolution2D(output_channel, kernel_size=(2, emb_size), padding='same',
                                 kernel_initializer=glorot_uniform_initializer, activation='relu')(emb_data)
    conv1_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, 500, output_channel)})(conv1_output)

    conv2_output = Convolution2D(output_channel, kernel_size=(4, emb_size), padding='same',
                                 kernel_initializer=glorot_uniform_initializer, activation='relu')(emb_data)
    conv2_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, 500, output_channel)})(conv2_output)

    conv3_output = Convolution2D(output_channel, kernel_size=(8, emb_size), padding='same',
                                 kernel_initializer=glorot_uniform_initializer, activation='relu')(emb_data)
    conv3_output = Lambda(reshape_tensor, arguments={'shape': (batch_size, 500, output_channel)})(conv3_output)

    pool1 = GlobalMaxPooling1D()(conv1_output)
    pool2 = GlobalMaxPooling1D()(conv2_output)
    pool3 = GlobalMaxPooling1D()(conv3_output)
    output = concatenate([pool1, pool2, pool3], axis=-1)
    output = Dense(3801, activation='softmax')(output)

    # model.add(Dense(num_bottleneck_hidden, input_shape=(ks * output_channel * dynamic_pool_length,), activation='relu',
    #                initializer=glorot_uniform_initializer))
    # model.add(Dropout(drop_out))
    # model.add(Dense(labels_num, input_shape=(num_bottleneck_hidden,), activation='relu',
    #                initializer=glorot_uniform_initializer))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(top_k=5)])
    # XMLCNN = Model
    model = Model(input_tensor, output)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(top_k=5)])
    # history = model.fit_generator(train_x, train_y.toArray(), epochs=epochs, batch_size=batch_size,
    #                               validation_data=(valid_x, valid_y.toArray()))
    model.fit_generator(steps_per_epoch=data_num / batch_size,
                        generator=batch_generator(train_x, train_y, batch_size),
                        nb_epoch=nb_epochs)

    model.save('keras_xmlcnn.h5')


if __name__ == '__main__':
    # print("torch cuda is available: ", torch.cuda.is_available())
    PROJECT_CONF = "E:/PycharmProject/CorNet/configure/"
    data_cnf = PROJECT_CONF + "datasets/EUR-Lex.yaml"
    model_cnf = PROJECT_CONF + "models/Keras-CorNetXMLCNN-EUR-Lex.yaml"
    mode = "train"
    main(data_cnf=data_cnf, model_cnf=model_cnf, mode=mode)
