import torch
import torch.nn as nn
import torchvision.models as models
import logging
from pytorch_transformers import BertModel, BertTokenizer
from pytorch_transformers import *
from typing import List


class ImgEncoder(nn.Module):

    def __init__(self, embed_size, pretrain_model):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        self.pretrain_model = pretrain_model

        super(ImgEncoder, self).__init__()
        if self.pretrain_model == 'vgg19':
            model = models.vgg19(pretrained=True)
            in_features = model.classifier[-1].in_features  # input size of feature vector
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1])    # remove last fc layer
            self.model = model                              # loaded model without last fc layer
            self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

        elif self.pretrain_model == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=True)
            in_features = model.classifier[-1].in_features  # input size of feature vector
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1])  # remove last fc layer
            self.model = model  # loaded model without last fc layer
            self.fc = nn.Linear(in_features, embed_size)  # feature vector of image

        elif self.pretrain_model == 'vgg16':
            model = models.vgg16(pretrained=True)
            in_features = model.classifier[-1].in_features  # input size of feature vector
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1])  # remove last fc layer
            self.model = model  # loaded model without last fc layer
            self.fc = nn.Linear(in_features, embed_size)  # feature vector of image

        elif self.pretrain_model == 'resnet34':
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            self.model = model
            self.fc = nn.Linear(num_ftrs, embed_size)

        elif self.pretrain_model == 'resnet152':
            model = models.resnet152(pretrained=True)
            num_ftrs = model.fc.in_features
            self.model = model
            self.fc = nn.Linear(num_ftrs, embed_size)


    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        if self.pretrain_model == 'vgg19' or 'vgg19_bn' or 'vgg16':
            img_feature = self.fc(img_feature)                   # [batch_size, embed_size]
            # this line is used only for VGG19

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)  # !!!for question: do word2vec here
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class BertEmbedder(nn.Module):

    def __init__(self, word_embed_size, num_layers, hidden_size, embed_size, max_seq_len=32):
        super().__init__()
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer_class = BertTokenizer
        self.model_class = BertModel
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.max_seq_len = max_seq_len
        self.model = self.model_class.from_pretrained(self.pretrained_weights)

        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states


    def forward(self, question):

      # needs [batch_size, max_qst_length=30, word_embed_size=300]
      qst_vec = self.model(question)[0]  # [batch_size, max_qst_length=30, word_embed_size=300]
      qst_vec = self.tanh(qst_vec)
      qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
      print("transpose:")
      print(qst_vec.size())

      _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
      qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
      print("cat:")
      print(qst_feature.size())

      qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
      print("transpose:")
      print(qst_feature.size())

      qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
      print("reshape:")
      print(qst_feature.size())

      qst_feature = self.tanh(qst_feature)
      print("tanh:")
      print(qst_feature.size())

      qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

      print("qst_feature:")
      print(qst_feature.size())

      return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, pretrain_model):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size, pretrain_model)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]

        print(img_feature.size())
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        print(qst_feature.size())

        # here we need combine bert feature
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature


class Bert_VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, pretrain_model):

        super(Bert_VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size, pretrain_model)
        self.bert_encoder = BertEmbedder(word_embed_size, num_layers, hidden_size, embed_size)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        print("img_feature:")
        print(img_feature.size())
        bert_qst_feature = self.bert_encoder(qst)
        print("bert_qst_feature:")
        print(bert_qst_feature.size())

        # here we need combine bert feature
        combined_feature = torch.mul(img_feature, bert_qst_feature)  # [batch_size, embed_size]

        combined_feature = self.tanh(combined_feature)

        combined_feature = self.dropout(combined_feature)

        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]

        combined_feature = self.tanh(combined_feature)

        combined_feature = self.dropout(combined_feature)

        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature

