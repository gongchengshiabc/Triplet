import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch.nn.functional as f
import os
from evaluate import l2norm
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder=models.vgg16(pretrained=True)
        self.decoder.classifier._modules['6'] = nn.Linear(4096, 256)
    def forward(self,x):
        feature=self.decoder(x)
        feature=l2norm(feature)
        return feature
class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.classify=nn.Linear(256,125)
    def forward(self,feature):
        classify=self.classify(feature)
        return classify
class SBIR(object):
    def __init__(self,opt):
        self.decoder=Decoder()
        self.classify=Classification()
        if torch.cuda.is_available():
            self.decoder.cuda()
            self.classify.cuda()
        #self.triplet_loss=ContrastiveLoss(opt.margin,opt.measure,opt.max_violation,opt.cost_style)
        self.classify_loss=nn.CrossEntropyLoss()
        self.triplet_loss=TripletLoss(1,1,opt.margin)
        parms=list(self.classify.parameters())+list(self.decoder.parameters())
        if opt.optimizer=='SGD':
            self.optimizer=optim.SGD([p for p in parms],lr=opt.learning_rate,momentum=opt.momentum,weight_decay=opt.weight_decay)
        elif opt.optimizer=='Rmspror':
            self.optimizer=optim.RMSprop([p for p in parms],lr=opt.learning_rate)
        elif opt.optimizer=='Adam':
            self.optimizer=optim.Adam([p for p in parms],lr=opt.learning_rate)

    def train_start(self):
        self.decoder.train()
        self.classify.train()
    def val_Start(self):
        self.decoder.eval()
        self.classify.eval()
    def forward_emd(self,photo):
        photo=Variable(photo)
        if torch.cuda.is_available():
            photo=photo.cuda()
        feature=self.decoder(photo)
        return feature
    def forward_classify(self,feature):
        feature= Variable(feature)
        if torch.cuda.is_available():
            feature = feature.cuda()
        classify = self.classify(feature)
        return classify
    def forward_loss(self,archor,positive,negative,input1,input2,input3,label):
        loss1=self.triplet_loss(archor,positive,negative)
        label = Variable(torch.from_numpy((np.array(list(map(int, label)))))).cuda()
        loss2=self.classify_loss(input1,label)+self.classify_loss(input2,label)+self.classify_loss(input3,label)
        loss=loss1+loss2
        return loss
    def train(self,archor,positive,negative,label):
        archor=self.forward_emd(archor)
        positive=self.forward_emd(positive)
        negative=self.forward_emd(negative)
        input1 = self.forward_classify(archor)
        input2 = self.forward_classify(positive)
        input3 = self.forward_classify(negative)
        self.optimizer.zero_grad()
        loss=self.forward_loss(archor,positive,negative,input1,input2,input3,label)
        loss_value = loss.data[0]
        loss.backward()
        self.optimizer.step()
        return loss_value
