import numpy as np
from common import Constants
import utils


class eval_batch:
    def __init__(self, tag_2_idx):
        self.tag_2_idx = tag_2_idx
        self.idx_2_tag = {v: k for k, v in self.tag_2_idx.items()}
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0
        self.totalp_counts = {}
        self.truep_counts = {}
        self.fn_counts = {}
        self.fp_counts = {}
        self.f1 = {}

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0
        self.totalp_counts = {}
        self.truep_counts = {}
        self.fn_counts = {}
        self.fp_counts = {}
        self.f1 = {}

    def f1_score(self):
        """
        calculate f1 score based on statics
        """
        print(f'overlap_count:{self.overlap_count}')
        print(f'guess_count:{self.guess_count}')
        print(f'gold_count:{self.gold_count}')

        if self.guess_count == 0:
            return {'total': (0.0, 0.0, 0.0, 0.0)}
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return {'total', (0.0, 0.0, 0.0, 0.0)}
        f = 2 * (precision * recall) / (precision + recall)
        accuracy = float(self.correct_labels) / self.total_labels
        message = ""
        self.f1['total'] = (f, precision, recall, accuracy, message)
        for label in self.totalp_counts:
            tp = self.truep_counts.get(label, 1)
            fn = sum(self.fn_counts.get(label, {}).values())
            fp = sum(self.fp_counts.get(label, {}).values())
            # print(label, str(tp), str(fp), str(fn), str(self.totalp_counts.get(label,0)))
            precision = tp / float(tp + fp + 1e-9)
            recall = tp / float(tp + fn + 1e-9)
            f = 2 * (precision * recall) / (precision + recall + 1e-9)
            message = str(self.fn_counts.get(label, {}))
            self.f1[label] = (f, precision, recall, 0, message)
        return self.f1

    def acc_score(self):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy

    def eval_instance(self, best_path, gold):
        """
        update statics for one instance
        args:
            best_path (seq_len): predicted
            gold (seq_len): ground-truth
        """

        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))
        for i in range(total_labels):
            gold_label = self.idx_2_tag[gold[i]]
            guessed_label = self.idx_2_tag[best_path[i]]
            self.totalp_counts[gold_label] = 1 + self.totalp_counts.get(gold_label, 0)
            if gold_label == guessed_label:
                self.truep_counts[gold_label] = 1 + self.truep_counts.get(gold_label, 0)
            else:
                val = self.fn_counts.get(gold_label, {})
                val[guessed_label] = 1 + val.get(guessed_label, 0)
                self.fn_counts[gold_label] = val

                val2 = self.fp_counts.get(guessed_label, {})
                val2[gold_label] = 1 + val2.get(gold_label, 0)
                self.fp_counts[guessed_label] = val2

        gold_chunks = utils.bio_to_chuncks(gold, self.idx_2_tag)
        gold_count = len(gold_chunks)

        guess_chunks = utils.bio_to_chuncks(best_path, self.idx_2_tag)
        guess_count = len(guess_chunks)

        # print(f'eva_matrix\n{gold_chunks}')
        # print(f'guess_chunks\n{guess_chunks}')
        # print('-'*10)

        overlap_count = 0
        overlap_chunks = []
        for gd_c in gold_chunks:
            for gs_c in guess_chunks:
                if gs_c.equals_name(gd_c):
                    overlap_count += 1
                    overlap_chunks.append(gs_c)
        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_f1_batch(self, best_path, gold):
        """
        update statics for f1 score
        args:
            best_path :Tensor(batch_size, seq_len): prediction sequence
            gold :Tensor(batch_size, seq_len): ground-truth
        """
        for decoded, target in zip(best_path, gold):
            # remove padding
            decoded = np.array(decoded)
            target = np.array(target)

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(
                decoded, target)
            self.correct_labels += correct_labels_i
            self.total_labels += total_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i

    def calc_acc_batch(self, best_path, gold):
        """
        update statics for accuracy
        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        for decoded, target in zip(best_path, gold):
            # remove padding
            decoded = np.array(decoded)
            target = np.array(target)

            self.total_labels += len(target)
            self.correct_labels += np.sum(np.equal(decoded, target))


class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)
    args:
        l_map: dictionary for labels
        score_type: use f1score with using 'f'
    """

    def __init__(self, tag_2_idx, score_type):
        eval_batch.__init__(self, tag_2_idx)

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, batch_guess_path, batch_gold_path):
        '''
        计算该批次的score
        :param batch_gold_path:[batch_sz,seq_len]
        :param batch_guess_path:[batch_sz,seq_len]
        :return:
        '''
        self.reset()
        self.eval_b(batch_guess_path, batch_gold_path)

        return self.calc_s()
