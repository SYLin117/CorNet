import torch
import torch.nn as nn
import torch.nn.functional as F

from deepxml.cornet_hsu import CorNet


class XMLCNN(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters,
                 vocab_size=None, emb_size=None, emb_trainable=True, emb_init=None, padding_idx=0, **kwargs):
        print("XMLCNN__init__")
        print("dropout = %s, labels_num = %s, dynamic_pool_length = %s, bottleneck_dim = %s"
              % (dropout, labels_num, dynamic_pool_length, bottleneck_dim))
        print("num_filters = %s, vocab_size = %s, emb_size = %s, emb_trainable = %s"
              % (num_filters, vocab_size, emb_size, emb_trainable))
        print("emb_init.shape = ", emb_init.shape, ", padding_idx = ", padding_idx)
        # dropout = 0.5, labels_num = 3801, dynamic_pool_length = 8, bottleneck_dim = 512
        # num_filters = 128, vocab_size = None, emb_size = 300, emb_trainable = False
        # emb_init.shape = (166402, 300), padding_idx = 0

        super(XMLCNN, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape  # (166402, 300)

        self.output_channel = num_filters
        self.num_bottleneck_hidden = bottleneck_dim
        self.dynamic_pool_length = dynamic_pool_length

        # num_embedding: 所有文本詞彙Index的數量
        # embedding_dim: 一個詞應該轉換成多少維度的向量
        # padding_idx: 如果有給數值，那麼在詞數不夠的情況下會使用你所設定的數值進行padding，讓每個輸入都維持同樣尺寸
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable

        self.ks = 3  # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        # 40 x 500 x 300
        self.conv1 = nn.Conv2d(1, self.output_channel, (2, emb_size), padding=(1, 0))
        self.conv2 = nn.Conv2d(1, self.output_channel, (4, emb_size), padding=(3, 0))
        self.conv3 = nn.Conv2d(1, self.output_channel, (8, emb_size), padding=(7, 0))
        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)  # Adaptive pooling

        self.bottleneck = nn.Linear(self.ks * self.output_channel * self.dynamic_pool_length,
                                    self.num_bottleneck_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, labels_num)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.bottleneck.weight)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        embe_out = self.emb(x)  # (batch, sent_len, embed_dim) # 40 x 500 x 300
        x = embe_out.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim) 增加維度
        # 40 x 1 x 500 x 300

        # (conv1): Conv2d(1, 128, kernel_size=(2, 300), stride=(1, 1), padding=(1, 0))
        # 40 x 128 x 500 x 1
        # (conv2): Conv2d(1, 128, kernel_size=(4, 300), stride=(1, 1), padding=(3, 0))
        # 40 x 128 x 500 x 1
        # (conv3): Conv2d(1, 128, kernel_size=(8, 300), stride=(1, 1), padding=(7, 0))
        # 40 x 128 x 500 x 1

        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]

        # [40 x (128 x 8)] x 3
        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        # ks = 3, output_channel = 128 , dynamic_pool_length = 8
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        # 40 x 512
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        # 40 x 3801
        return logit


class CorNetXMLCNN(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters, **kwargs):
        # dropout = 0.5, labels_num = 3801, dynamic_pool_length = 8, bottleneck_dim = 512
        # num_filters = 128, vocab_size = None, emb_size = 300, emb_trainable = False
        # emb_init.shape = (166402, 300), padding_idx = 0
        super(CorNetXMLCNN, self).__init__()
        self.xmlcnn = XMLCNN(dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters, **kwargs)
        self.cornet = CorNet(labels_num, **kwargs)

    def forward(self, input_variables):
        raw_logits = self.xmlcnn(input_variables)
        cor_logits = self.cornet(raw_logits)
        return cor_logits