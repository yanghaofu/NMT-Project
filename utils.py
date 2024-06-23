import os
import logging
import sentencepiece as spm


def chinese_tokenizer_load():
    """
    加载中文分词模型。

    这个函数加载一个 SentencePiece 中文分词模型，该模型存储在 'tokenizer/chn.model' 文件中。
    通过 SentencePieceProcessor 类来加载模型，并返回处理器对象。

    返回：
        sp_chn (SentencePieceProcessor): 已加载的中文分词模型处理器。
    """
    sp_chn = spm.SentencePieceProcessor()
    sp_chn.Load('{}.model'.format("tokenizer/chn"))
    return sp_chn


def english_tokenizer_load():
    """
    加载英文分词模型。

    这个函数加载一个 SentencePiece 英文分词模型，该模型存储在 'tokenizer/eng.model' 文件中。
    通过 SentencePieceProcessor 类来加载模型，并返回处理器对象。

    返回：
        sp_eng (SentencePieceProcessor): 已加载的英文分词模型处理器。
    """
    sp_eng = spm.SentencePieceProcessor()
    sp_eng.Load('{}.model'.format("tokenizer/eng"))
    return sp_eng


def set_logger(log_path):
    """
    设置日志记录器，将日志信息记录到终端和指定的文件中。

    这个函数设置一个日志记录器，用于将信息记录到终端和指定的日志文件中。
    如果指定的日志文件已存在，则会先删除该文件。
    日志记录器会记录所有级别为 INFO 及以上的日志信息。

    参数：
        log_path (str): 日志文件的路径。
    """
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志记录器的级别为 INFO

    if not logger.handlers:
        # 创建文件处理器，将日志记录到指定文件
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # 创建控制台处理器，将日志记录到终端
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
