import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import chinese_tokenizer_load
import torch.nn.functional as F

def calculate_r_dbr(attn_weights):
    """计算差异性正则化项 R_DBR。

    参数:
    attn_weights -- 注意力分布矩阵，形状为 (batch_size, num_heads, seq_len, seq_len)

    返回:
    r_dbr -- 差异性正则化项
    """
    h = attn_weights.size(1)  # 注意力头的数量
    r_dbr = 0.0  # 初始化差异性正则化项

    # 遍历所有的注意力头
    for i in range(h):
        for j in range(h):
            if i != j:  # 只计算不同注意力头之间的KL散度
                # 计算第 i 个注意力头和第 j 个注意力头之间的KL散度
                r_dbr += F.kl_div(attn_weights[:, i], attn_weights[:, j], reduction='batchmean')

    # 平均化KL散度
    r_dbr = -r_dbr / (h * (h - 1))  # 归一化为最终的差异性正则化项并取负值
    return r_dbr


def calculate_scr(attn_weights):
    """
    计算空间一致性正则化项 R_SCR
    参数：
    attn_weights: 注意力权重矩阵，形状为 (batch_size, num_heads, seq_len, seq_len)

    返回：
    scr: 空间一致性正则化项
    """
    batch_size, num_heads, seq_len, _ = attn_weights.size()
    avg_attn_weights = attn_weights.mean(dim=1)  # 计算平均注意力权重
    avg_attn_weights = avg_attn_weights.view(batch_size, seq_len, seq_len)

    # 将注意力权重映射到空间上
    G = avg_attn_weights  # 假设注意力权重已经是空间形式

    # 找到最大值的位置
    max_idx = G.view(batch_size, -1).argmax(dim=1)
    max_i = max_idx // seq_len
    max_j = max_idx % seq_len

    # 计算 R_SCR
    scr = 0.0
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                r = ((max_i[b] - i) ** 2 + (max_j[b] - j) ** 2) ** 0.5
                scr += torch.exp(-1.0 * r) * torch.norm(G[b, max_i[b], max_j[b]] - G[b, i, j])

    return scr / (batch_size * seq_len * seq_len)

def run_epoch(data, model, loss_compute):
    """
    运行一个训练或验证周期

    参数:
    - data: 数据集，包含多个批次的数据
    - model: 待训练或验证的模型
    - loss_compute: 计算损失并进行反向传播的函数

    返回:
    - 平均损失
    """
    total_tokens = 0  # 初始化总token数
    total_loss = 0  # 初始化总损失

    # 遍历数据集中的每个批次
    with tqdm(data, unit="batch") as tepoch:
        for batch in tepoch:
            # 前向传播计算模型输出
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            # 计算当前批次的损失
            loss = loss_compute(out, batch.trg_y, batch.ntokens)

            # # 获取注意力权重
            # enc_attn_weights, dec_attn_weights = model.module.get_attention_weights()
            #
            # # 计算差异性正则化项并加入总损失
            # for attn_weights in enc_attn_weights + dec_attn_weights:
            #     if attn_weights is not None:
            #         r_dbr = calculate_r_dbr(attn_weights)
            #         loss += r_dbr

            # 累加当前批次的损失和token数
            total_loss += loss
            total_tokens += batch.ntokens

            # 更新进度条描述
            tepoch.set_postfix(loss=(total_loss / total_tokens).item())
        torch.cuda.empty_cache()

    # 返回平均损失
    return total_loss / total_tokens


def create_random_subset(dataset, subset_size):
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices)


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        # 创建随机子集

        dev_subset = create_random_subset(dev_data, 100)
        dev_dataloader = DataLoader(dev_subset, shuffle=False, batch_size=config.batch_size,
                                    collate_fn=dev_data.collate_fn)

        model.eval()
        dev_loss = run_epoch(dev_dataloader, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(dev_dataloader, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """一个用于多GPU损失计算和训练的函数类"""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # 将生成器和损失函数分发到不同的GPU上
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0  # 初始化总损失为0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)  # 复制生成器到各个GPU
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)  # 将输出分散到各个GPU
        out_grad = [[] for _ in out_scatter]  # 初始化存储梯度的列表
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)  # 将目标值分散到各个GPU

        # 将生成的输出分块处理
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # 预测分布
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算损失
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # 累加并归一化损失
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # 将损失反向传播到Transformer的输出
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # 将所有损失反向传播到Transformer中
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize  # 返回总损失乘以归一化因子


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + trg[i] + '|||' + res[i] + '\n'
                fp.write(line)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=True):
    """
    用训练好的模型进行单句预测，打印模型翻译结果

    参数:
    - src: 输入的源语言句子张量
    - model: 已训练好的Transformer模型
    - use_beam: 是否使用Beam Search进行解码，默认为True

    返回:
    - 打印翻译结果
    """
    # 加载中文分词器
    sp_chn = chinese_tokenizer_load()

    # 使用torch.no_grad()上下文管理器，禁用梯度计算
    with torch.no_grad():
        # 加载训练好的模型参数
        model.load_state_dict(torch.load(config.model_path))
        # 设置模型为评估模式
        model.eval()

        # 对输入的源语言句子进行掩码操作，掩码的维度是(1, 1, 源语言句子长度)
        src_mask = (src != 0).unsqueeze(-2)

        # 判断是否使用Beam Search解码
        if use_beam:
            # 使用Beam Search进行解码，获取解码结果和分数
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            # 只取每个解码结果的第一个序列
            decode_result = [h[0] for h in decode_result]
        else:
            # 使用贪婪解码进行解码，获取解码结果
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)

        # 将解码结果的ID转换为中文字符
        translation = [sp_chn.decode_ids(_s) for _s in decode_result]
        # 打印翻译结果
        print(translation[0])
