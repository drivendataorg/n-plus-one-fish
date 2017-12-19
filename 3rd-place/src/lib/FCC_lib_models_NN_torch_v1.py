from torch import np # Torch wrapper for Numpy

import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import functional as F


import sys
import time
import copy


#####
#     METRICS & TOOLS
#####


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, per_class=False):
    r"""Function that measures Binary Cross Entropy between target and output
    logits:

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch.
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    if weight is not None and target.dim() != 1:
        weight = weight.view(1, target.size(1)).expand_as(target)

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

    if weight is not None:
        loss = loss * weight
        
    if per_class:
        return loss.mean(0)
    
    if size_average:
        return loss.mean()
    else:
        return loss.sum()
        
class BCEWithLogitsLoss(Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    This Binary Cross Entropy between the target and the output logits
    (no sigmoid applied) is:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    or in the case of the weights argument being specified:

    .. math:: loss(o, t) = - 1/n \sum_i weights[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    By default, the losses are averaged for each minibatch over observations
    *as well as* over dimensions. However, if the field `size_average` is set
    to `False`, the losses are instead summed.

    """
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average)
        else:
            return binary_cross_entropy_with_logits(input, target, size_average=self.size_average)

                     
#####
#     torch_wrapper
#####

class NNtorch_v0(object):
    
    def  __init__(self, torch_model, criterion=None, optimizers=None, use_cuda=True):
        self.torch_model = torch_model
        self.criterion = criterion
        self.metrics_names =  criterion.__str__()
        if not isinstance(optimizers, list):
            optimizers = [optimizers, ]
        self.optimizers = optimizers
        self.use_cuda = use_cuda
        self.model_in_gpu = False
        
    
    def set_cuda_option(self, use_cuda):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.torch_model = self.torch_model.cuda()
            self.criterion = self.criterion.cuda()
            self.model_in_gpu = True
        else:
            self.torch_model = self.torch_model.cpu()
            self.criterion = self.criterion.cpu()
            self.model_in_gpu = False
    
    def save_model(self, filename):
        torch_model = copy.deepcopy(self.torch_model)
        torch_model = torch_model.cpu()
        torch.save(torch_model, filename)
    
    def load_model(self, filename):
        self.torch_model = torch.load(filename)
        if self.use_cuda:
            self.torch_model = self.torch_model.cuda()

    def train_loader(self, data_loader, epochs, initial_epoch, steps_per_epoch,
              lr_schedulers=None, args_dict={}):
        
        reduce_labels = args_dict.get('reduce_labels', False)
        
        #init
        since = time.time()
        if self.use_cuda and (~self.model_in_gpu):
            self.set_cuda_option(self.use_cuda)
        optimizers = self.optimizers
        lr_schedulers = [None,] * len(optimizers) if lr_schedulers is None else lr_schedulers
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers, ]
        num_epochs = epochs - initial_epoch
        
        
        for epoch in range(initial_epoch, epochs):
            since_epoch = time.time()
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-' * 10)
            
            ### Train
            # Update optimizer
            for i in range(len(optimizers)):
                if lr_schedulers[i] is not None:
                    optimizers[i] = lr_schedulers[i](optimizers[i], epoch)
            _ = self.torch_model.train(True)  # Set model to training mode

            # Iterate over data
            running_loss = 0.0
            count_batches = 0
            count_samples = 0
            total_batches = len(data_loader)
            batch_time = time.time()
            for data in data_loader:
                # get the inputs
                inputs, labels = data
                if reduce_labels:
                    _, labels = torch.max(labels, dim=1)
                    labels = labels[:,0]
                    
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                count_batches += 1
                count_samples += inputs.size(0)
                
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels) 
                
                # zero the parameter gradients
                for i in range(len(optimizers)):
                    optimizers[i].zero_grad()
                
                # forward
                outputs = self.torch_model(inputs)
                loss = self.criterion(outputs, labels)
                
                # loss calculations
                batch_loss = loss.data[0]
                running_loss += batch_loss * inputs.size(0)
                loss_sofar = running_loss / count_samples
                
                # backward + optimize 
                loss.backward()
                for i in range(len(optimizers)):
                    optimizers[i].step()

                # Print
                if (time.time() - batch_time) > 2 or count_batches >= total_batches:
                    if count_batches > 1:
                        sys.stdout.write ('\r')
                        sys.stdout.flush()
                    sys.stdout.flush()    
                    sys.stdout.write('Batches:{}/{} Samples:{} Loss: {:.4f}'.\
                                     format(count_batches, total_batches, count_samples, loss_sofar))
                    sys.stdout.flush()
                    batch_time = time.time()

            print('')
        
            time_elapsed = time.time() - since_epoch
            print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            print('')
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def evaluate_loader(self, data_loader, args_dict={}):
        
        reduce_labels = args_dict.get('reduce_labels', False)
        
        #init
        since = time.time()
        if self.use_cuda and (~self.model_in_gpu):
            self.set_cuda_option(self.use_cuda)
            
        ### Evaluate
        _ = self.torch_model.train(False)  # Set model to evaluation mode
        _ = self.torch_model.eval()

        # Iterate over data
        running_loss = 0.0
        count_batches = 0
        count_samples = 0
        total_batches = len(data_loader)
        batch_time = time.time()
        for data in data_loader:
            # get the inputs
            inputs, labels = data
            if reduce_labels:
                    _, labels = torch.max(labels, dim=1)
                    labels = labels[:,0]
                    
            if self.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            count_batches += 1
            count_samples += inputs.size(0)
            
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels) 
            
            # forward
            outputs = self.torch_model(inputs)
            loss = self.criterion(outputs, labels)
            
            # loss calculations
            batch_loss = loss.data[0]
            running_loss += batch_loss * inputs.size(0)
            loss_sofar = running_loss / count_samples

            # Print
            if (time.time() - batch_time) > 2 or count_batches >= total_batches:
                if count_batches > 1:
                    sys.stdout.write ('\r')
                    sys.stdout.flush()
                sys.stdout.flush()    
                sys.stdout.write('Batches:{}/{} Samples:{} Loss: {:.4f}'.\
                                 format(count_batches, total_batches, count_samples, loss_sofar))
                sys.stdout.flush()
                batch_time = time.time()

        print('')

        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        return [loss_sofar,]

    def predict_loader(self, data_loader, verbose=False):
        
        #init
        since = time.time()
        if self.use_cuda and (~self.model_in_gpu):
            self.set_cuda_option(self.use_cuda)
        pred = []
            
        ### Evaluate
        _ = self.torch_model.train(False)  # Set model to evaluation mode
        _ = self.torch_model.eval()

        # Iterate over data
        count_batches = 0
        count_samples = 0
        total_batches = len(data_loader)
        batch_time = time.time()
        for data in data_loader:
            # get the inputs
            inputs, _ = data
            if self.use_cuda:
                inputs = inputs.cuda()
            count_batches += 1
            count_samples += inputs.size(0)
            
            # wrap them in Variable
            inputs = Variable(inputs, volatile=True)
            
            # forward
            outputs = self.torch_model(inputs)
            pred.append(outputs.data.cpu().numpy())

            # Print
            if (time.time() - batch_time) > 2 or count_batches >= total_batches:
                if count_batches > 1:
                    sys.stdout.write ('\r')
                    sys.stdout.flush()
                sys.stdout.flush()    
                sys.stdout.write('Batches:{}/{} Samples:{}'.format(count_batches, total_batches, count_samples))
                sys.stdout.flush()
                batch_time = time.time()

        pred = np.vstack(pred)

        time_elapsed = time.time() - since
        if verbose:
            print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        return pred
    
    def predict(self, batch, verbose=False):
        #init
        since = time.time()
        if self.use_cuda and (~self.model_in_gpu):
            self.set_cuda_option(self.use_cuda)
            
        ### Evaluate
        _ = self.torch_model.train(False)  # Set model to evaluation mode
        _ = self.torch_model.eval()

        # Predict
        # get the inputs
        inputs = torch.from_numpy(batch.astype(np.float32))
        if self.use_cuda:
            inputs = inputs.cuda()
            
        # wrap them in Variable
        inputs = Variable(inputs, volatile=True)
            
        # forward
        outputs = self.torch_model(inputs)
        pred = outputs.data.cpu().numpy()

        time_elapsed = time.time() - since
        if verbose:
            print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        return pred   
    

#####
#     PRETRAINED MODELS
#####

def pretrain_vgg11_v0(channels, isz, classes, args_dict={}):
    from torchvision import models
    from collections import OrderedDict  
    import copy
    
    interm = args_dict.get('interm', 4096)

    PTmodel = models.vgg11(pretrained=True)
    PTmodel.classifier = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(25088, interm)),
            ('fc1.relu', torch.nn.ReLU(inplace=True)),
            
            ('fc2.drop', torch.nn.Dropout(0.50)),
            ('fc2', torch.nn.Linear(interm, interm)),
            ('fc2.relu', torch.nn.ReLU(inplace=True)),
            
            ('fc3.drop', torch.nn.Dropout(0.50)),
            ('fc3', torch.nn.Linear(interm, classes)),
            ]))
    torch_model = copy.deepcopy(PTmodel)

    criterion = args_dict.get('criterion', torch.nn.BCELoss())
    
    optimizer = torch.optim.SGD([{'params': torch_model.features.parameters(), 'lr':0.00},
                       {'params': torch_model.classifier.parameters(), 'lr':0.01},
                       ],
                      momentum = 0.9)
    optimizers = args_dict.get('optimizers', optimizer)
    
    NNmodel = NNtorch_v0(torch_model, criterion, optimizers, use_cuda=False)
    return NNmodel

def pretrain_vgg16_v0(channels, isz, classes, args_dict={}):
    from torchvision import models
    from collections import OrderedDict  
    import copy
    
    interm = args_dict.get('interm', 4096)

    PTmodel = models.vgg16(pretrained=True)
    PTmodel.classifier = torch.nn.Sequential(OrderedDict([
            #('fc1.drop', torch.nn.Dropout(0.50)),
            ('fc1', torch.nn.Linear(25088, interm)),
            ('fc1.relu', torch.nn.ReLU(inplace=True)),
            
            ('fc2.drop', torch.nn.Dropout(0.50)),
            ('fc2', torch.nn.Linear(interm, interm)),
            ('fc2.relu', torch.nn.ReLU(inplace=True)),
            
            ('fc3.drop', torch.nn.Dropout(0.50)),
            ('fc3', torch.nn.Linear(interm, classes)),
            ]))
    torch_model = copy.deepcopy(PTmodel)

    criterion = args_dict.get('criterion', torch.nn.BCELoss())
    
    optimizer = torch.optim.SGD([{'params': torch_model.features.parameters(), 'lr':0.00},
                       {'params': torch_model.classifier.parameters(), 'lr':0.01},
                       ],
                      momentum = 0.9)
    optimizers = args_dict.get('optimizers', optimizer)
    
    NNmodel = NNtorch_v0(torch_model, criterion, optimizers, use_cuda=False)
    return NNmodel
