import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    """实现词嵌入层"""

    def __init__(self, d_model, vocab):
        """
        初始化Embedding类

        参数:
        - d_model: 词嵌入的维度
        - vocab: 词汇表的大小
        """
        super(Embeddings, self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # 记录词嵌入的维度
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播计算词嵌入

        参数:
        - x: 输入的词汇索引序列

        返回:
        - 对应的词嵌入矩阵，乘以math.sqrt(d_model)进行缩放
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        """
        形式如：
        tensor([[0.],
                [1.],
                [2.],
                [3.],
                [4.],
                ...])
        """
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    实现局部增强的注意力机制，增强局部性特征。

    参数:
    - query: 查询矩阵，形状为 (batch_size, num_heads, seq_len, d_k)
    - key: 键矩阵，形状为 (batch_size, num_heads, seq_len, d_k)
    - value: 值矩阵，形状为 (batch_size, num_heads, seq_len, d_v)
    - window_size: 窗口大小，定义每个query主要关注其前后window_size个位置的key和value
    - mask: 掩码矩阵（可选）
    - dropout: Dropout层（可选）

    返回:
    - 注意力权重加权后的value矩阵
    - 注意力矩阵
    """
    window_size = 4
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 创建局部增强的掩码矩阵
    batch_size, num_heads, seq_len, _ = scores.size()
    # print(seq_len)
    local_mask = torch.zeros_like(scores)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        local_mask[:, :, i, start:end] = 1

    # 对注意力得分进行加权
    enhanced_scores = scores + (scores * local_mask)

    if mask is not None:
        enhanced_scores = enhanced_scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(enhanced_scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# def attention(query, key, value, mask=None, dropout=None):
#     """
#     计算注意力机制
#
#     参数:
#     - query: 查询矩阵
#     - key: 键矩阵
#     - value: 值矩阵
#     - mask: 掩码矩阵（可选）
#     - dropout: Dropout层（可选）
#
#     返回:
#     - 注意力权重加权后的value矩阵
#     - 注意力矩阵
#     """
#     # 将query矩阵的最后一个维度值作为d_k
#     d_k = query.size(-1)
#
#     # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#
#     # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#
#     # 将mask后的attention矩阵按照最后一个维度进行softmax
#     p_attn = F.softmax(scores, dim=-1)
#
#     # 如果dropout参数设置为非空，则进行dropout操作
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
#     return torch.matmul(p_attn, value), p_attn


# def attention(query, key, value, mask=None, dropout=None):
#     """
#     计算注意力机制，包含高斯偏差
#
#     参数:
#     - query: 查询矩阵
#     - key: 键矩阵
#     - value: 值矩阵
#     - mask: 掩码矩阵（可选）
#     - dropout: Dropout层（可选）
#
#     返回:
#     - 注意力权重加权后的value矩阵
#     - 注意力矩阵
#     """
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#
#     # 计算高斯偏差，将scores的最大值作为P_i
#     P_i = torch.max(scores, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
#     D_i = scores.size(-1)/2  # 最后一维的长度
#     sigma_i = D_i / 2
#
#     # 计算G
#     G = (scores - P_i) ** 2 / (2 * sigma_i ** 2)
#
#     # 创建一个与G相同形状的零矩阵
#     zero_matrix = torch.zeros_like(G)
#     # 将G的前两维度填充为0
#     G[:, :, :, :] = zero_matrix[:, :, :, :]
#
#     # 加入高斯偏差到注意力得分
#     scores += G
#
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#
#     p_attn = F.softmax(scores, dim=-1)
#
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#
#     return torch.matmul(p_attn, value), p_attn


# 多头注意力机制的实现
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化多头注意力机制类

        参数:
        - h: 注意力头的数量
        - d_model: 词嵌入的维度
        - dropout: Dropout的概率
        """
        super(MultiHeadedAttention, self).__init__()
        # 确保d_model可以被h整除
        assert d_model % h == 0
        # 计算每个头的维度
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # 定义4个全连接层，分别用于WQ, WK, WV矩阵和最终的输出变换矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 初始化注意力权重矩阵
        self.attn = None
        # 初始化保存注意力权重
        self.attn_weights = None
        # 定义Dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播计算多头注意力

        参数:
        - query: 查询矩阵
        - key: 键矩阵
        - value: 值矩阵
        - mask: 掩码矩阵（可选）

        返回:
        - 多头注意力的输出
        """
        if mask is not None:
            # 在第二个维度上增加一个维度
            mask = mask.unsqueeze(1)
        # 获取batch大小
        nbatches = query.size(0)
        # 将query, key, value分别通过对应的全连接层，得到新的表示
        # 然后将结果拆成h块，并将第二个和第三个维度值互换
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 计算注意力，得到注意力矩阵和加权后的value
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 保存注意力权重
        self.attn_weights = self.attn
        # 注意力得分 self.attn 是一个形状为 (batch_size, num_heads, seq_len, seq_len) 的矩阵，表示每个头的注意力权重。
        # 将多头注意力矩阵concat起来，并将h变回到第三维的位置
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 通过最后一个全连接层变换后返回
        return self.linears[-1](x)

    def get_attn_weights(self):
        """获取注意力权重矩阵"""
        return self.attn_weights

    def calculate_r_dbr(self):
        """计算差异性正则化项 R_DBR。

        返回:
        r_dbr -- 差异性正则化项
        """
        h = self.h  # 注意力头的数量
        r_dbr = 0.0  # 初始化差异性正则化项

        # 遍历所有的注意力头
        for i in range(h):
            for j in range(h):
                if i != j:  # 只计算不同注意力头之间的KL散度
                    # 计算第 i 个注意力头和第 j 个注意力头之间的KL散度
                    r_dbr += F.kl_div(self.attn_weights[:, i], self.attn_weights[:, j], reduction='batchmean')

        # 平均化KL散度
        return r_dbr / (h * (h - 1))  # 归一化为最终的差异性正则化项


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 返回Layer Norm的结果
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 第一个线性层，将输入维度从d_model变为d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二个线性层，将维度从d_ff变回d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 首先通过第一个线性层和ReLU激活函数
        x = self.w_1(x)
        x = F.relu(x)
        # 然后通过Dropout层
        x = self.dropout(x)
        # 最后通过第二个线性层
        x = self.w_2(x)
        return x


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # 自注意力机制
        self.self_attn = self_attn
        # 前馈神经网络
        self.feed_forward = feed_forward
        # 子层连接，包含两个子层，分别是自注意力机制和前馈神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 模型尺寸
        self.size = size

    def forward(self, x, mask):
        # 通过自注意力机制层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 通过前馈神经网络层
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # 模型的尺寸
        self.size = size
        # 自注意力机制
        self.self_attn = self_attn
        # 与编码器输出进行的注意力机制
        self.src_attn = src_attn
        # 前馈神经网络
        self.feed_forward = feed_forward
        # 包含三个子层连接，分别是自注意力机制、上下文注意力机制和前馈神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放编码器的最终hidden表示结果
        m = memory

        # 自注意力机制：注意自注意力机制的query、key和value均为解码器hidden状态
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 上下文注意力机制：注意上下文注意力机制的query为解码器hidden状态，而key和value为编码器hidden状态
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 前馈神经网络
        return self.sublayer[2](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def get_attention_weights(self):
        """获取编码器和解码器中的注意力权重"""
        enc_attn_weights = [layer.self_attn.get_attn_weights() for layer in self.encoder.layers]
        dec_attn_weights = [layer.self_attn.get_attn_weights() for layer in self.decoder.layers]
        return enc_attn_weights, dec_attn_weights


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    创建Transformer模型

    参数:
    - src_vocab: 源语言词汇表的大小
    - tgt_vocab: 目标语言词汇表的大小
    - N: 编码器和解码器层的数量
    - d_model: 词嵌入的维度
    - d_ff: 前馈神经网络的维度
    - h: 多头注意力机制中的头数
    - dropout: Dropout的概率

    返回:
    - 初始化后的Transformer模型
    """
    c = copy.deepcopy
    # 实例化多头注意力对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化位置前馈神经网络对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化位置编码对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # 初始化参数，使用Glorot / fan_avg方法
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """
    使用贪婪解码策略生成目标句子序列

    参数:
    - model: 已训练好的Transformer模型
    - src: 输入的源语言句子张量
    - src_mask: 源语言句子的掩码张量
    - max_len: 解码的最大长度
    - start_symbol: 起始符号的ID
    - end_symbol: 结束符号的ID

    返回:
    - results: 解码得到的目标句子序列
    """
    # 获取batch的大小和源句子的序列长度
    batch_size, src_seq_len = src.size()
    # 初始化结果列表，每个batch中的序列对应一个空列表
    results = [[] for _ in range(batch_size)]
    # 初始化停止标志列表，每个batch中的序列对应一个False
    stop_flag = [False for _ in range(batch_size)]
    # 初始化已结束序列的计数
    count = 0

    # 使用编码器对源语言句子进行编码
    memory = model.encode(src, src_mask)
    # 初始化目标序列张量，填充起始符号
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    # 进行解码循环，最大解码长度为max_len
    for s in range(max_len):
        # 生成目标序列的掩码，并扩展为与batch大小一致
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
        # 使用解码器进行解码，获取输出
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        # 使用生成器将输出转换为词汇表大小的概率分布
        prob = model.generator(out[:, -1, :])
        # 获取最大概率的词ID作为预测结果
        pred = torch.argmax(prob, dim=-1)

        # 将预测结果拼接到目标序列中
        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    # 如果预测结果为结束符号，则标记停止
                    count += 1
                    stop_flag[i] = True
                else:
                    # 否则，将预测结果添加到对应的结果列表中
                    results[i].append(pred[i].item())
            if count == batch_size:
                # 如果所有序列都已结束解码，则退出循环
                break

    return results


def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """传入一个训练好的模型，对指定数据进行预测"""
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
