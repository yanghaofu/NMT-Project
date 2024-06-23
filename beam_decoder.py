import torch
from data_loader import subsequent_mask


class Beam:
    """ Beam search """

    def __init__(self, size, pad, bos, eos, device=False):
        """
        初始化Beam搜索类

        参数:
        - size: Beam大小，即保留的候选序列数量
        - pad: PAD符号的ID
        - bos: BOS符号的ID
        - eos: EOS符号的ID
        - device: 是否使用GPU
        """
        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # 初始化每个候选序列的得分为0
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # 存储每个时间步的回溯指针
        self.prev_ks = []

        # 存储每个时间步的输出
        # 初始化为 [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """获取当前时间步的输出"""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """获取当前时间步的回溯指针"""
        return self.prev_ks[-1]

    @property
    def done(self):
        """判断Beam搜索是否完成"""
        return self._done

    def advance(self, word_logprob):
        """
        更新Beam状态，并检查是否完成

        参数:
        - word_logprob: 当前时间步每个词的对数概率
        """
        num_words = word_logprob.size(1)

        # 累加之前的得分
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # 初始情况
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId 被展平为一个 (beam x word) 数组，
        # 因此我们需要计算每个得分来自哪个词和beam
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # 当Beam顶部为EOS时结束
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """对得分进行排序"""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """获取Beam中得分最高的序列及其得分"""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """获取当前时间步的解码序列"""

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ 回溯构建完整的假设 """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))



def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    """ 在一个批次中进行翻译 """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ 获取实例在张量中的位置。 """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """ 收集与活动实例相关的张量部分。 """
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # 活动实例数 * Beam大小 x 序列长度 x 维度
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # 选择仍然活跃的张量部分
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        """ 收集活动实例信息 """
        # 收集仍然活跃的句子，以便解码器不会在已完成的句子上运行
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        """ 解码并更新Beam状态，然后返回活动Beam的索引 """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            """ 准备解码序列 """
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # Batch大小 x Beam大小 x 解码序列长度
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # Batch大小*Beam大小 x 解码序列长度
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            """ 预测下一个词 """
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            out = model.decode(enc_output, src_mask, dec_seq, subsequent_mask(dec_seq.size(1)).type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            """ 收集活跃实例的索引列表 """
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # 获取每个Beam的解码序列
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # 获取每个Beam的词概率
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)

        # 更新Beam并收集未完成的实例
        active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        """ 收集假设和得分 """
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # 编码
        src_enc = model.encode(src, src_mask)

        # 为Beam搜索重复数据
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        # 准备Beams
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # 活跃实例的簿记
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # 解码
        for len_dec_seq in range(1, max_len + 1):
            active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break  # 所有实例都已完成其路径到<EOS>

            # 过滤掉不活跃的张量部分
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)

    return batch_hyp, batch_scores
