#!/usr/bin/env python
# coding: utf-8

from Focal_Loss import focal_loss
import torch

pred = torch.randn((3,5))
print("pred:",pred)

label = torch.tensor([2,3,4])
print("label:",label)


# alpha设定为0.25,对第一类影响进行减弱(目标检测任务中,第一类为背景类)
loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)

# alpha输入列表,分别对每一类施加不同的权重
loss_fn = focal_loss(alpha=[1,2,3,1,2], gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)


# GPU调用
pred = pred.to('cuda')
print("pred:",pred)

label = label.to('cuda')
print("label:",label)
loss_fn = focal_loss(alpha=[1,2,3,1,2], gamma=2, num_classes=5)
loss = loss_fn(pred, label)
print(loss)

