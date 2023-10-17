# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch
import torch.nn as nn
import numpy as np

# from layers import *
from ofa.layers import set_layer_from_config, MBInvertedConvLayer, ConvLayer, IdentityLayer, LinearLayer
from ofa.imagenet_codebase.utils import MyNetwork, make_divisible
from ofa.imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock


class MobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier):

        super(MobileNetV3, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

from ofa.elastic_nn.modules.dynamic_layers import ExitBlock

class EEMobileNetV3(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, feature_mix_layer, classifier, 
    n_classes, dropout_rate, d_list, t):

        super(EEMobileNetV3, self).__init__()

        self.base_stage_width = [24, 40, 80, 112]
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        self.feature_mix_layer = feature_mix_layer
        self.classifier = classifier
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.d_list = d_list
        self.threshold = t

        #set_threshold t_list
        #self.mask = t != 1 ## mask with 1 for exit and 0 for non-exit
        exit_idxs = []
        exit_list = []
        t_list = []
        n_blocks = len(self.base_stage_width)+1
        idx = 1
        for i in range(0,n_blocks-1,1):
            idx += self.d_list[i]
            if (t[i]!=1):
                t_list.append(t[i])
                feature_dim = [self.base_stage_width[i]]
                final_expand_width = [feature_dim[0] * 6] #960
                last_channel = [feature_dim[0] * 8] #1280
                exit_idxs.append(idx)
                exit_list.append(ExitBlock(self.n_classes,final_expand_width,feature_dim,last_channel,self.dropout_rate))
        self.n_exit = len(exit_list)
        self.t_list = t_list
        self.exit_idxs = exit_idxs
        self.exit_list = nn.ModuleList(exit_list)

    def forward(self, x):

        x = self.first_conv(x)

        preds = [] 
        idxs = []

        if(self.training): #training 
            i = 0
            for idx,block in enumerate(self.blocks):
                if(self.n_exit!=0):
                    if (idx==self.exit_idxs[i]): #exit block
                        pred, _ = self.exit_list[i](x)
                        preds.append(pred)
                        if(i<(self.n_exit-1)):
                            i+=1
                x = block(x)
            x = self.final_expand_layer(x)
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
            x = self.feature_mix_layer(x)
            x = torch.squeeze(x)
            x = self.classifier(x)
            preds.append(x)
            return preds
        else:
            i = 0
            counts = np.zeros(self.n_exit+1)
            for idx,block in enumerate(self.blocks):
                if(self.n_exit!=0):
                    if (idx==self.exit_idxs[i]): #exit block
                            exit_block = self.exit_list[i]
                            # move network to GPU if available
                            if torch.cuda.is_available():
                                device = torch.device('cuda:0')
                                exit_block.to(device)
                            pred, conf = exit_block(x)
                            conf = torch.squeeze(conf)
                            mask = conf >= self.t_list[i]
                            mask = mask.cpu() #gpu>cpu memory
                            p = np.where(np.array(mask)==False)[0] #idxs of non EE predictions
                            counts[i] = torch.sum(mask).item()
                            if (x.shape[0]==1): #if batch size = 1
                                if mask.item()==1: #exit
                                    x = torch.empty(0,x.shape[1],x.shape[2],x.shape[3])
                                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                                    x = x.to(device)
                                else:
                                    pred = torch.empty(0,pred.shape[0])
                            else:
                                x = x[mask==0,:,:,:]
                                pred = pred[mask==1,:]
                            del mask 
                            del conf
                            if(pred.size(-1) != self.n_classes and pred.numel() != 0): #(pred.dim()!=0): #if not empty tensor
                                print("ANOMALY: pred.shape = ",pred.shape)
                            preds.append(pred)
                            idxs.append(p)
                            # FIX bug that for one sample x.shape = (0,1,,,,) when empty
                            if(i<(self.n_exit-1)):
                                i+=1
                x = block(x)

            counts[-1] = x.shape[0] #n samples classified normally by the last exit
            x = self.final_expand_layer(x)
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
            x = self.feature_mix_layer(x)
            x = torch.squeeze(x)
            x = self.classifier(x)

            if(self.n_exit!=0):

                preds.append(x)

                for pred in preds:
                    if(pred.size(-1) != self.n_classes and pred.numel() != 0): #(pred.dim()!=0): #if not empty tensor
                                    print("ANOMALY3: pred.shape = ",pred.shape)

                #mix predictions of all exits
                tensors = []

                for i in range(len(preds)-1,0,-1): #mix predictions of all exits
                    tensors = list(torch.unbind(preds[i-1],axis=0)) #preds of the previous exit

                    filtered_tensors = []
                    for tensor in tensors:
                        if tensor.numel() != self.n_classes and tensor.numel() != 0:
                            print("ANOMALY2: tensor.shape = ",tensor)
                        else:
                            filtered_tensors.append(tensor)
                    tensors = filtered_tensors

                    iter = idxs[i-1]  
                    pred = preds[i]
                    for j,idx in enumerate(iter):
                        if pred[j].numel() > 0: #(pred[j].dim()!=0): #if not empty tensor
                          tensors.insert(idx,pred[j])
                    
                    if tensors:
                        preds[i-1] = torch.stack(tensors,axis=0)

                    del preds[i]

                '''
                for i in range(len(preds)-1,0,-1): #mix predictions of all exits
                    tensors = list(torch.unbind(preds[i-1],axis=0)) #preds of the previous exit

                    iter = idxs[i-1]  
                    pred = preds[i]
                    for j,idx in enumerate(iter):
                        if pred[j].numel() == self.n_classes:#(pred[j].dim()!=0): #if not empty tensor
                          tensors.insert(idx,pred[j])

                    filtered_tensors = []
                    for tensor in tensors:
                        if tensor.numel() == self.n_classes:
                            filtered_tensors.append(tensor)
                    tensors = filtered_tensors
                    
                    if tensors:
                        #print(tensors)
                        preds[i-1] = torch.stack(tensors,axis=0)
                    #else:
                    #    preds[i-1] = torch.empty(0,preds[i-1].shape[1])

                    del preds[i]
                '''
                
                x = preds[0]
                del preds[0]
                del pred
            
            return x,counts
        

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str
        return _str

    @property
    def config(self):
        return {
            'name': MobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            'feature_mix_layer': self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        first_conv = set_layer_from_config(config['first_conv'])
        final_expand_layer = set_layer_from_config(config['final_expand_layer'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        classifier = set_layer_from_config(config['classifier'])

        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        net = EEMobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net
    
    def set_threshold(self,t):
        exit_idxs = []
        exit_list = []
        t_list = []
        n_blocks = len(self.base_stage_width)+1
        idx = 1
        for i in range(0,n_blocks-1,1):
            idx += self.d_list[i]
            if (t[i]!=1):
                t_list.append(t[i])
                feature_dim = [self.base_stage_width[i]]
                final_expand_width = [feature_dim[0] * 6] #960
                last_channel = [feature_dim[0] * 8] #1280
                exit_idxs.append(idx)
                exit_list.append(ExitBlock(self.n_classes,final_expand_width,feature_dim,last_channel,self.dropout_rate))
        self.n_exit = len(exit_list)
        self.t_list = t_list
        self.exit_idxs = exit_idxs
        self.exit_list = nn.ModuleList(exit_list)

    def zero_last_gamma(self):
        for m in self.modules():
            if isinstance(m, MobileInvertedResidualBlock):
                if isinstance(m.mobile_inverted_conv, MBInvertedConvLayer) and isinstance(m.shortcut, IdentityLayer):
                    m.mobile_inverted_conv.point_linear.bn.weight.data.zero_()

    @staticmethod
    def build_net_via_cfg(cfg, input_channel, last_channel, n_classes, dropout_rate):
        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='h_swish', ops_order='weight_bn_act'
        )
        # build mobile blocks
        feature_dim = input_channel
        blocks = []
        for stage_id, block_config_list in cfg.items():
            for k, mid_channel, out_channel, use_se, act_func, stride, expand_ratio in block_config_list:
                mb_conv = MBInvertedConvLayer(
                    feature_dim, out_channel, k, stride, expand_ratio, mid_channel, act_func, use_se
                )
                if stride == 1 and out_channel == feature_dim:
                    shortcut = IdentityLayer(out_channel, out_channel)
                else:
                    shortcut = None
                blocks.append(MobileInvertedResidualBlock(mb_conv, shortcut))
                feature_dim = out_channel
        # final expand layer
        final_expand_layer = ConvLayer(
            feature_dim, feature_dim * 6, kernel_size=1, use_bn=True, act_func='h_swish', ops_order='weight_bn_act',
        )
        feature_dim = feature_dim * 6
        # feature mix layer
        feature_mix_layer = ConvLayer(
            feature_dim, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
        )
        # classifier
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        return first_conv, blocks, final_expand_layer, feature_mix_layer, classifier

    @staticmethod
    def adjust_cfg(cfg, ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        for i, (stage_id, block_config_list) in enumerate(cfg.items()):
            for block_config in block_config_list:
                if ks is not None and stage_id != '0':
                    block_config[0] = ks
                if expand_ratio is not None and stage_id != '0':
                    block_config[-1] = expand_ratio
                    block_config[1] = None
                    if stage_width_list is not None:
                        block_config[2] = stage_width_list[i]
            if depth_param is not None and stage_id != '0':
                new_block_config_list = [block_config_list[0]]
                new_block_config_list += [copy.deepcopy(block_config_list[-1]) for _ in range(depth_param - 1)]
                cfg[stage_id] = new_block_config_list
        return cfg

class MobileNetV3Large(MobileNetV3):

    def __init__(self, n_classes=1000, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0.2,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None):
        input_channel = 16
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult, 8)
        last_channel = make_divisible(last_channel * width_mult, 8) if width_mult > 1.0 else last_channel

        cfg = {
            #    k,     exp,    c,      se,         nl,         s,      e,
            '0': [
                [3,     16,     16,     False,      'relu',     1,      1],
            ],
            '1': [
                [3,     64,     24,     False,      'relu',     2,      None],  # 4
                [3,     72,     24,     False,      'relu',     1,      None],  # 3
            ],
            '2': [
                [5,     72,     40,     True,       'relu',     2,      None],  # 3
                [5,     120,    40,     True,       'relu',     1,      None],  # 3
                [5,     120,    40,     True,       'relu',     1,      None],  # 3
            ],
            '3': [
                [3,     240,    80,     False,      'h_swish',  2,      None],  # 6
                [3,     200,    80,     False,      'h_swish',  1,      None],  # 2.5
                [3,     184,    80,     False,      'h_swish',  1,      None],  # 2.3
                [3,     184,    80,     False,      'h_swish',  1,      None],  # 2.3
            ],
            '4': [
                [3,     480,    112,    True,       'h_swish',  1,      None],  # 6
                [3,     672,    112,    True,       'h_swish',  1,      None],  # 6
            ],
            '5': [
                [5,     672,    160,    True,       'h_swish',  2,      None],  # 6
                [5,     960,    160,    True,       'h_swish',  1,      None],  # 6
                [5,     960,    160,    True,       'h_swish',  1,      None],  # 6
            ]
        }

        cfg = self.adjust_cfg(cfg, ks, expand_ratio, depth_param, stage_width_list)
        # width multiplier on mobile setting, change `exp: 1` and `c: 2`
        for stage_id, block_config_list in cfg.items():
            for block_config in block_config_list:
                if block_config[1] is not None:
                    block_config[1] = make_divisible(block_config[1] * width_mult, 8)
                block_config[2] = make_divisible(block_config[2] * width_mult, 8)

        first_conv, blocks, final_expand_layer, feature_mix_layer, classifier = self.build_net_via_cfg(
            cfg, input_channel, last_channel, n_classes, dropout_rate
        )
        super(MobileNetV3Large, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
