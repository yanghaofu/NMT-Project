import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from train import train, test, translate
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing


class NoamOpt:
    """实现学习率调度的优化器包装类。"""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer  # 优化器实例
        self._step = 0  # 当前步数
        self.warmup = warmup  # 预热步数
        self.factor = factor  # 调整因子
        self.model_size = model_size  # 模型尺寸
        self._rate = 0  # 当前学习率

    def step(self):
        """更新参数和学习率"""
        self._step += 1  # 更新步数
        rate = self.rate()  # 计算新的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 设置优化器中的学习率
        self._rate = rate  # 更新当前学习率
        self.optimizer.step()  # 进行优化器的步进

    def rate(self, step=None):
        """计算学习率"""
        if step is None:
            step = self._step  # 使用当前步数
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """为批量大小为32，5530步数为一个epoch，2个epoch用于预热的模型获取标准优化器"""
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# 定义函数来创建子集
def create_subset(dataset, subset_size):
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices)

def run():
    utils.set_logger(config.log_path)  # 设置日志记录

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    print(len(train_dataset), len(dev_dataset), len(test_dataset))

    logging.info("-------- 数据集构建完成! --------")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    # dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
    #                             collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- 获取数据加载器完成! --------")
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)  # 并行化模型
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataset, dev_dataset, model, model_par, criterion, optimizer)
    print("训练完成!")
    test(test_dataloader, model, criterion)


def check_opt():
    """检查学习率变化"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # 三种学习率超参数设置
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent, beam_search=True):
    """单句翻译"""
    # 初始化模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    BOS = english_tokenizer_load().bos_id()  # 开始标记
    EOS = english_tokenizer_load().eos_id()  # 结束标记
    src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)


def translate_example():
    """单句翻译示例"""
    sent = "I am the Newton of this century."
    # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
    one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    import warnings
    warnings.filterwarnings('ignore')
    run()
    # translate_example()
