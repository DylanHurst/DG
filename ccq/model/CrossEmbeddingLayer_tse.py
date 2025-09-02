import torch
import torch.nn as nn
import torch.nn.functional as F

##################
# 测试
##################
import model.my_repeat_padding_atten_value
from model.my_repeat_padding_atten_value import transform_matrix
from model.my_extract_repeat_atten_rank import select_indices
#################
import torch
import MyTensorSaver

def replace_rows_batch(matrix, k_vec):
    # 获取矩阵的批次数量 batch_size、行数 rows 和列数 cols
    batch_size, rows, cols = matrix.size()

    # 初始化一个列表来存储每个矩阵的结果
    result_tensor_list = []

    # 对每个批次的矩阵进行处理
    for i in range(batch_size):
        k = k_vec[i]  # 对于每个批次，使用对应的 k 值

        # 保留每个矩阵的第0行和第k-1行不变
        result_tensor = matrix[i, :k, :]  # 先把前k行保留到结果中

        # 创建一个新tensor，存储每个矩阵从第1行到第k-2行的元素
        replacement_rows = matrix[i, 1:k - 1, :]

        # 循环直到矩阵的最后一行
        while result_tensor.size(0) < rows:
            # 从replacement_rows中循环取出元素进行替换
            for row in replacement_rows:
                if result_tensor.size(0) < rows:
                    result_tensor = torch.cat((result_tensor, row.unsqueeze(0)), dim=0)

        result_tensor_list.append(result_tensor)

    # 将结果列表转换成一个 tensor
    return torch.stack(result_tensor_list)


def count_nonzero_per_row(matrix):
    # 对矩阵进行非零判断，返回一个布尔矩阵，值为True表示非零元素
    nonzero_matrix = matrix != 0

    # 对每行的非零元素进行统计，沿着列的方向求和，得到每行非零元素个数
    nonzero_count = nonzero_matrix.sum(dim=1, keepdim=True)  # keepdim=True 保证返回的结果为 (m, 1)

    return nonzero_count  # 返回一个形状为 (m, 1) 的张量，表示每行非零元素的个数


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


class LinearProjection(nn.Module):
    def __init__(self, input_rows=146, output_rows=5, feature_dim=512):
        super().__init__()
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.feature_dim = feature_dim

        # 定义线性层，将 146 行映射到 5 行
        self.linear = nn.Linear(input_rows, output_rows)

    def forward(self, x):
        # 输入 x 的形状为 (146, 512)
        # 转置为 (512, 146)，以便线性层对行数进行操作
        x = x.transpose(0, 1)  # 形状变为 (512, 146)

        # 通过线性层将 146 行映射到 5 行
        x = self.linear(x)  # 形状变为 (512, 5)

        # 转置回 (5, 512)
        x = x.transpose(0, 1)  # 形状变为 (5, 512)

        return x


import torch
import torch.nn as nn


class DynamicLinearProjection(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x, output_rows):
        # 输入 x 的形状为 (input_rows, feature_dim)
        input_rows, feature_dim = x.size()

        # 动态创建线性层，将 input_rows 映射到 output_rows
        linear = nn.Linear(input_rows, output_rows).to(x.device)

        # 转置为 (feature_dim, input_rows)，以便线性层对行数进行操作
        x = x.transpose(0, 1)  # 形状变为 (feature_dim, input_rows)
        linear = linear.half()
        # 通过线性层将 input_rows 映射到 output_rows
        x = linear(x)  # 形状变为 (feature_dim, output_rows)

        # 转置回 (output_rows, feature_dim)
        x = x.transpose(0, 1)  # 形状变为 (output_rows, feature_dim)

        return x


import torch
import torch.nn as nn

import torch.nn.functional as F


# 只能固定几个模型，自适应获取，不能预测具体哪一个是1，2，3，4 ，只能临时判断，临时获取，这样就可以学习了
# 随机只有一次就是在定义模型结构的时候。
class DynamicLinearProjection_32_512_to_5_1024(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024, k=96, max_rows=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.max_rows = max_rows

        # 主线性层（用于 output_rows > 8 的情况）
        self.linear = nn.Linear(input_dim, output_dim).half()

        # 动态生成线性层和参数
        self.linears = nn.ModuleDict()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        for i in range(1, max_rows + 1):
            # 动态创建线性层：linear_i (k*i -> output_dim)
            self.linears[f'linear_{i}'] = nn.Linear(k * i, output_dim).half()

            # 动态创建权重和偏置 (i x input_dim)
            self.weights[f'weight_{i}'] = nn.Parameter(torch.randn(i, input_dim))
            self.biases[f'bias_{i}'] = nn.Parameter(torch.zeros(i))

    def forward(self, x, output_rows):
        """
        输入 x 的形状: (batch_size, input_dim)
        output_rows: 决定使用的权重和线性层
        """
        if 1 <= output_rows <= self.max_rows:
            # 动态获取权重、偏置和线性层
            weight = self.weights[f'weight_{output_rows}'].half()
            bias = self.biases[f'bias_{output_rows}'].half()
            linear = self.linears[f'linear_{output_rows}']

            # 统一处理
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
        self.max_dynamic_rows = 128  # 动态生成的最大行数

        # 主线性层（用于output_rows > max_rows的情况）
        self.linear = nn.Linear(input_dim, output_dim).half()

        # 使用ModuleDict和ParameterDict动态存储
        self.linears = nn.ModuleDict()
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        # 动态生成线性层和参数
        for i in range(1, max_rows + 1):
            self.linears[f'linear_{i}'] = nn.Linear(k * i, output_dim).half()
            self.weights[f'weight_{i}'] = nn.Parameter(torch.randn(i, input_dim))
            self.biases[f'bias_{i}'] = nn.Parameter(torch.zeros(i))

    def forward(self, x, output_rows):
        # 输入验证
        if x.size(1) != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.size(1)}")

        if not (1 <= output_rows <= self.max_dynamic_rows):
            raise ValueError(f"output_rows must be between 1 and {self.max_dynamic_rows}, got {output_rows}")

        # 处理1-max_rows的情况
        if output_rows <= self.max_rows:
            weight = self.weights[f'weight_{output_rows}'].half()
            bias = self.biases[f'bias_{output_rows}'].half()
            linear = self.linears[f'linear_{output_rows}']

            x = F.linear(x, weight, bias).transpose(0, 1)  # (M, input_dim)
            return linear(x)  # (M, output_dim)

        # 处理max_rows+1到128的情况
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

class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=2048, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio
        self.dynamicLinearProjection = TXT_DynamicLinearProjection_32_512_to_5_1024(input_dim=512, output_dim=2048, k=int(30), max_rows=MyTensorSaver.pidstatistics)

    def forward(self, features, text, atten, pid):
        # 确保输入数据在 GPU 上
        features = features.to('cuda')
        text = text.to('cuda')
        atten = atten.to('cuda')
        pid = pid.to('cuda')

        # 计算 mask 和 lengths
        mask = (text != 0).float()
        lengths = mask.sum(1) - 2  # -2 for SOS and EOS tokens
        k = int((atten.size(1) - 2) * 0.5)
        k = 30
        bs = features.size(0)

        # 处理 atten
        atten[torch.arange(bs), :, text.argmax(dim=-1)] = -1  # last token
        atten[torch.arange(bs), :, 0] = -1  # first token
        atten = atten[torch.arange(bs), text.argmax(dim=-1), :]  # 64 x 77
        atten = atten * mask

        # 提取 top-k 注意力特征
        # @!!@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!注释掉了这两行，因为不需要排序，反而影响原来内容
        atten_topK = atten.topk(dim=-1, k=k)[1].unsqueeze(-1).expand(bs, k, features.size(2))  # 64 x k x 512
        features = torch.gather(input=features, dim=1, index=atten_topK)  # 64 x k x 512

        # 生成 Different_features_page
        # @!!@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!!!!注释掉了这两行，因为不需要排序，反而影响原来内容
        # Different_features_page = [features[i, :lengths[i].int(), :] for i in range(bs)]
        Different_features_page = features
        # 计算 labels 和 mask
        pid = pid.reshape((bs, 1))  # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()
        mask = 1 - labels

        # 分组操作
        mask_row_sum = (1 - mask).sum(1)
        indices = (mask_row_sum >= 1).nonzero(as_tuple=True)[0]
        zero_positions = [(idx.item(), ((mask)[idx] == 0).nonzero(as_tuple=True)[0].tolist()) for idx in indices]

        # 去重分组
        seen = set()
        result = []
        for item in zero_positions:
            list_as_tuple = tuple(item[1])
            if list_as_tuple not in seen:
                seen.add(list_as_tuple)
                result.append(item)

        # 分组特征拼接和动态线性投影
        features_cat_page_group = []
        for col_num, many_text_is_one_pid in result:
            selected_matrices = [Different_features_page[i] for i in many_text_is_one_pid]
            features_cat_page_group.append(torch.cat(selected_matrices, dim=0))

        # 动态线性投影
        #  for i in range(len(result)):
        #    if features_cat_page_group[i].size(0)==6 or features_cat_page_group[i].size(1)==450:
        #   dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for i in range(len(result))]
        dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for
                                         i in range(len(result))]
        New_base_features = torch.cat(dynamicLinearProjection_group, dim=0)

        # 创建一个全零张量，用于存储最终的结果
        # 假设最终输出的行数为 32，列数为 512
        output = torch.zeros(New_base_features.size(0), 2048).to('cuda')

        # 遍历 result 和 dynamicLinearProjection_group
        for (k, indices), tensor in zip(result, dynamicLinearProjection_group):
            # 将 dynamicLinearProjection_group[k] 的行恢复到 result[k][1] 中的位置
            for i, idx in enumerate(indices):
                output[idx] = tensor[i]  # 将 tensor 的第 i 行赋值给 output 的第 idx 行
        New_base_features = output

        # 归一化和 MLP
        features = l2norm(features, dim=-1)
        lengths = torch.Tensor([min(lengths[i].item(), k) for i in range(bs)]).to(features.device)
        New_base_features = l2norm(New_base_features, dim=-1)
        features = self.mlp(features)  # 残差连接
        features = maxk_pool1d_var(features, 1, 1, lengths)  # max pooling
        features = features + New_base_features

        return features.float()


# import MyTensorSaver
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=2048, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.dynamicLinearProjection = DynamicLinearProjection_32_512_to_5_1024(input_dim=512, output_dim=2048, k=int(192*0.5),max_rows=MyTensorSaver.pidstatistics)

    def forward(self, base_features, atten, pid):
        # 确保输入数据在 GPU 上
        base_features = base_features.to('cuda')
        atten = atten.to('cuda')
        pid = pid.to('cuda')

        # 计算 k
        k = int((atten.size(1) - 1) * 0.5)  # k=(193-1)*0.3 = 57(整型）

        # 处理 atten
        bs = base_features.size(0)  # bs=64,第0维尺寸，代表batch_size
        atten[torch.arange(bs), :, 0] = -1  # CLS token=0行和0列所有元素，atten：（64,193，193），局部atten:(64,192,192)
        atten_topK = atten[:, 0].topk(dim=-1, k=k)[
            1]  # 局部注意力只取一行大到小排序，只取64张图片的，第0行，里面有全局注意力向量值，取这个全局注意力向量的最大前57个数值，每张图片都是，最后就是（64，57）尺寸，相当于每个注意力矩阵取57个0行数值
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k,
                                                     base_features.size(2))  # 64 x k x 512 将一行局部注意力扩展为一页特征尺寸与features一致
        base_features = torch.gather(input=base_features, dim=1, index=atten_topK)  # 64 x k x 512 按照最大注意力顺序提取对应位置的特征

        NoDifferent_features_page = base_features  # 相同的页，文本每页行数不等，但是图像相等

        # 获取标签，发现多个文本对应一个pid行人（可能有多张pid行人图片，因为有的数据集最大只有2个文本描述一个图片，有的时候，出现多个文本对应一个pid)
        batch_size = base_features.size(0)
        pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
        pid_dist = pid - pid.t()
        labels = (pid_dist == 0).float()
        mask = 1 - labels

        # 分组操作
        mask_row_sum = (1 - mask).sum(1)
        indices = (mask_row_sum >= 1).nonzero(as_tuple=True)[0]
        zero_positions = [(idx.item(), ((mask)[idx] == 0).nonzero(as_tuple=True)[0].tolist()) for idx in indices]

        # 去重分组
        seen = set()
        result = []
        for item in zero_positions:
            list_as_tuple = tuple(item[1])
            if list_as_tuple not in seen:
                seen.add(list_as_tuple)
                result.append(item)

        # 分组特征拼接和动态线性投影
        features_cat_page_group = []
        for col_num, many_text_is_one_pid in result:
            selected_matrices = [NoDifferent_features_page[i] for i in many_text_is_one_pid]
            features_cat_page_group.append(torch.cat(selected_matrices, dim=0))

        # 动态线性投影
        dynamicLinearProjection_group = [self.dynamicLinearProjection(features_cat_page_group[i], len(result[i][1])) for
                                         i in range(len(result))]
        New_base_features = torch.cat(dynamicLinearProjection_group, dim=0)

        # 创建一个全零张量，用于存储最终的结果
        # 假设最终输出的行数为 32，列数为 512
        output = torch.zeros(New_base_features.size(0), 2048).to('cuda')

        # 遍历 result 和 dynamicLinearProjection_group
        for (k, indices), tensor in zip(result, dynamicLinearProjection_group):
            # 将 dynamicLinearProjection_group[k] 的行恢复到 result[k][1] 中的位置
            for i, idx in enumerate(indices):
                output[idx] = tensor[i]  # 将 tensor 的第 i 行赋值给 output 的第 idx 行
        New_base_features = output

        # 归一化和 MLP
        base_features = l2norm(base_features, dim=-1)
        base_features = base_features.half()
        feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device).half()
        feat_lengths[:] = base_features.size(1)

        New_base_features = l2norm(New_base_features, dim=-1)
        features = self.mlp(base_features)  # 残差连接
        features = maxk_pool1d_var(features, 1, 1, feat_lengths)  # max pooling

        # 根据条件添加 New_base_features
        # if MyTensorSaver.count_num == 2:
        # features = features + New_base_features
        features = features + New_base_features
        return features.float()

