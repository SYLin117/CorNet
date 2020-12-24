import torch
import torch.nn as nn
import torch.nn.functional as F

from deepxml.cornet import CorNet
from torchsummary import summary


class XMLCNN(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters,
                 vocab_size=None, emb_size=None, emb_trainable=True, emb_init=None, padding_idx=0, **kwargs):
        super(XMLCNN, self).__init__()
        if emb_init is not None:  # emb_init為2d array,(vocab_num, emb_size)
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape

        self.output_channel = num_filters
        self.num_bottleneck_hidden = bottleneck_dim
        self.dynamic_pool_length = dynamic_pool_length

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx, sparse=True,
                                _weight=torch.from_numpy(emb_init).float() if emb_init is not None else None)
        self.emb.weight.requires_grad = emb_trainable

        self.ks = 3  # There are three conv nets here
        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(1, self.output_channel, (2, emb_size), padding=(
            1, 0))  # in_channel, out_channel, kernel_size,padding:第一個引數表示高度padding,第二個引數為左右padding
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
        # print("tendor size:", x.size())
        x = x.long()
        print("x size:", x.size())
        embe_out = self.emb(x)  # (batch, sent_len, embed_dim)
        print("embe_out size:", embe_out.size())
        x = embe_out.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        print("unsqueeze embe_out size:", x.size())
        tmp1 = F.relu(self.conv1(x)).squeeze(3)
        tmp2 = F.relu(self.conv2(x)).squeeze(3)
        tmp3 = F.relu(self.conv3(x)).squeeze(3)
        print("tmp1 size:",tmp1.size())
        print("tmp2 size:", tmp2.size())
        print("tmp3 size:", tmp3.size())
        x = [tmp1, tmp2, tmp3]
        y = []
        for i in x:
            tmp = self.pool(i).squeeze(2)
            print("tmp size:",tmp.size())
            y.append(tmp)
        x = y

        # (batch, channel_output) * ks
        x = torch.cat(x, 1)  # (batch, channel_output * ks)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        x = self.dropout(x)
        logit = self.fc1(x)  # (batch, target_size)
        return logit


class CorNetXMLCNN(nn.Module):
    def __init__(self, dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters, **kwargs):
        super(CorNetXMLCNN, self).__init__()
        self.xmlcnn = XMLCNN(dropout, labels_num, dynamic_pool_length, bottleneck_dim, num_filters, **kwargs)
        self.cornet = CorNet(labels_num, **kwargs)

    def forward(self, input_variables):
        raw_logits = self.xmlcnn(input_variables)
        cor_logits = self.cornet(raw_logits)
        return cor_logits
