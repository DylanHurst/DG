import torch
import torch.nn as nn
import torch.nn.functional as F
from model.extract_repeat_atten_rank import select_indices
#################
import torch
import TensorSaver

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


import torch
import torch.nn as nn


class DynamicLinearProjection(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, output_rows):

        input_rows, feature_dim = x.size()
        linear = nn.Linear(input_rows, output_rows).to(x.device)
        x = x.transpose(0, 1)  #  (feature_dim, input_rows)
        linear = linear.half()

        x = linear(x)
        x = x.transpose(0, 1)  #  (output_rows, feature_dim)

        return x


import torch
import torch.nn as nn

import torch.nn.functional as F

class DynamicLinearProjection_32_512_to_5_1024(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, k=96, max_rows=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.max_rows = max_rows


        self.linear = nn.Linear(input_dim, output_dim).half()


        self.linears = nn.ModuleDict()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        for i in range(1, max_rows + 1):
            self.linears[f'linear_{i}'] = nn.Linear(k * i, output_dim).half()

            self.weights[f'weight_{i}'] = nn.Parameter(torch.randn(i, input_dim))
            self.biases[f'bias_{i}'] = nn.Parameter(torch.zeros(i))

    def forward(self, x, output_rows):

        if 1 <= output_rows <= self.max_rows:
            weight = self.weights[f'weight_{output_rows}'].half()
            bias = self.biases[f'bias_{output_rows}'].half()
            linear = self.linears[f'linear_{output_rows}']

            x = F.linear(x, weight, bias).transpose(0, 1)  # (M, input_dim)
            x = linear(x)  # (M, output_dim)
            return x

        elif output_rows > 8 and output_rows <= 128:
            x = x.half()
            x = x.transpose(0, 1)  # (input_dim, N)
            weight = torch.randn(output_rows, x.size(1), dtype=torch.float16, device=x.device)
            bias = torch.zeros(output_rows, dtype=torch.float16, device=x.device)
            x = F.linear(x, weight, bias).transpose(0, 1)  # (M, input_dim)
            x = self.linear(x)  # (M, output_dim)
            return x

        else:
            raise ValueError(f"output_rows must be between 1 and 128, got {output_rows}")


class TXT_DynamicLinearProjection_32_512_to_5_1024(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, k=75, max_rows=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.max_rows = max_rows
        self.max_dynamic_rows = 128  #

        self.linear = nn.Linear(input_dim, output_dim).half()


        self.linears = nn.ModuleDict()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()


        for i in range(1, max_rows + 1):
            self.linears[f'linear_{i}'] = nn.Linear(k * i, output_dim).half()
            self.weights[f'weight_{i}'] = nn.Parameter(torch.randn(i, input_dim))
            self.biases[f'bias_{i}'] = nn.Parameter(torch.zeros(i))

    def forward(self, x, output_rows):

        if x.size(1) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.size(1)}")

        if not (1 <= output_rows <= self.max_dynamic_rows):
            raise ValueError(f"output_rows must be between 1 and {self.max_dynamic_rows}, got {output_rows}")


        if output_rows <= self.max_rows:
            weight = self.weights[f'weight_{output_rows}'].half()
            bias = self.biases[f'bias_{output_rows}'].half()
            linear = self.linears[f'linear_{output_rows}']

            x = F.linear(x, weight, bias).transpose(0, 1)  # (M, input_dim)
            return linear(x)  # (M, output_dim)


        else:
            x = x.half().transpose(0, 1)  # (input_dim, N)
            weight = torch.randn(
                output_rows, x.size(1),
                dtype=torch.float16,
                device=x.device
            )
            bias = torch.zeros(
                output_rows,
                dtype=torch.float16,
                device=x.device
            )
            x = F.linear(x, weight, bias).transpose(0, 1)  # (M, input_dim)
            return self.linear(x)  # (M, output_dim)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

#text processing
class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=2048, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio
        self.dynamicLinearProjection = TXT_DynamicLinearProjection_32_512_to_5_1024(input_dim=512, output_dim=2048, k=int(30),max_rows=TensorSaver.pidstatistics)

    def forward(self, features, text, atten, pid):

        features = features.to('cuda')
        text = text.to('cuda')
        atten = atten.to('cuda')
        pid = pid.to('cuda')


        mask = (text != 0).float()
        lengths = mask.sum(1) - 2  # -2 for SOS and EOS tokens
        k = int((atten.size(1) - 2) * 0.4)
        k = 30
        bs = features.size(0)


        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1  # last token
        atten[torch.arange(bs), :, 0] = -1  # first token
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :]  # 64 x 77
        atten = atten * mask


        atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2))  # 64 x k x 512
        features = torch.gather(input=features, dim=1, index=atten_topK)  # 64 x k x 512


        # Different_features_page = [features[i, :lengths[i].int(), :] for i in range(bs)]
        Different_features_page = features

        pid = pid.reshape((bs, 1))  # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()
        mask = 1 - labels


        mask_row_sum = (1 - mask).sum(1)
        indices = (mask_row_sum >= 1).nonzero(as_tuple=True)[0]
        zero_positions = [(idx.item(), ((mask)[idx] == 0).nonzero(as_tuple=True)[0].tolist()) for idx in indices]


        seen = set()
        result = []
        for item in zero_positions:
            list_as_tuple = tuple(item[1])
            if list_as_tuple not in seen:
                seen.add(list_as_tuple)
                result.append(item)


        features_cat_page_group = []
        for col_num, many_text_is_one_pid in result:
            selected_matrices = [Different_features_page[i] for i in many_text_is_one_pid]
            features_cat_page_group.append(torch.cat(selected_matrices, dim=0))


        #  for i in range(len(result)):
        #    if features_cat_page_group[i].size(0)==6 or features_cat_page_group[i].size(1)==450:
        #   dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for i in range(len(result))]
        dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for
                                         i in range(len(result))]
        New_base_features = torch.cat(dynamicLinearProjection_group, dim=0)


        output = torch.zeros(New_base_features.size(0), 2048).to('cuda')

        # result and dynamicLinearProjection_group
        for (k, indices), tensor in zip(result, dynamicLinearProjection_group):
            for i, idx in enumerate(indices):
                output[idx] = tensor[i]
        New_base_features = output

        features = l2norm(features, dim=-1)
        lengths = torch.Tensor([min(lengths[i].item(), k) for i in range(bs)]).to(features.device)
        New_base_features = l2norm(New_base_features, dim=-1)
        features = self.mlp(features)  # mlp
        features = maxk_pool1d_var(features, 1, 1, lengths)
        features = features + New_base_features
        #features =  New_base_features
        return features.float()


# import TensorSaver
import torch
import torch.nn as nn
import torch.nn.functional as F

#image processing
class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=2048, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.dynamicLinearProjection = DynamicLinearProjection_32_512_to_5_1024(input_dim=512, output_dim=2048, k=int(192*0.5),max_rows=TensorSaver.pidstatistics)

    def forward(self, base_features, atten, pid):

        base_features = base_features.to('cuda')
        atten = atten.to('cuda')
        pid = pid.to('cuda')


        k = int((atten.size(1) - 1) * 0.5)  # k=(193-1)*0.3 = 57

        bs = base_features.size(0)  # bs=64,
        atten[torch.arange(bs), :, 0] = -1  # CLS
        atten_topK = atten[:, 0].topk(dim=-1, k=k)[1]
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k,base_features.size(2))
        base_features = torch.gather(input=base_features, dim=1, index=atten_topK)

        NoDifferent_features_page = base_features  #

        #
        batch_size = base_features.size(0)
        pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()
        mask = 1 - labels


        mask_row_sum = (1 - mask).sum(1)
        indices = (mask_row_sum >= 1).nonzero(as_tuple=True)[0]
        zero_positions = [(idx.item(), ((mask)[idx] == 0).nonzero(as_tuple=True)[0].tolist()) for idx in indices]
        seen = set()
        result = []
        for item in zero_positions:
            list_as_tuple = tuple(item[1])
            if list_as_tuple not in seen:
                seen.add(list_as_tuple)
                result.append(item)

        features_cat_page_group = []
        for col_num, many_text_is_one_pid in result:
            selected_matrices = [NoDifferent_features_page[i] for i in many_text_is_one_pid]
            features_cat_page_group.append(torch.cat(selected_matrices, dim=0))

        # 动态线性投影
        dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for
                                         i in range(len(result))]
        New_base_features = torch.cat(dynamicLinearProjection_group, dim=0)


        output = torch.zeros(New_base_features.size(0), 2048).to('cuda')


        for (k, indices), tensor in zip(result, dynamicLinearProjection_group):

            for i, idx in enumerate(indices):
                output[idx] = tensor[i]
        New_base_features = output

        # 归一化和 MLP
        base_features = l2norm(base_features, dim=-1)
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)

        New_base_features = l2norm(New_base_features, dim=-1)
        features = self.mlp(base_features)
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)  # max pooling

        # if TensorSaver.count_num == 2:
        features = features + New_base_features
        #features =  New_base_features
        return features.float()

