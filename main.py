import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
import math
import random

import config
import data_loader
import utils
from ner_model import Model as FeatureModel
from a2c_model import A2CModel
from ner_model_origin import Model as NERModel

import timeit

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                # data_batch = [data.cuda() for data in data_batch[:-1]]
                data_batch = [data.cpu() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, a2c_mask = data_batch

                outputs = ner_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.cpu().numpy(),
                                                             pred_result.cpu().numpy(),
                                                             average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # cpu load


def training_a2c(a2c_model, feature_model, data_loader, episodes, gamma, time_to_sample, coeff, optimizer, print_loss):

    for j in range(episodes):
        print('epoch:', j)
        for i, data_batch in enumerate(data_loader):
            entity_text = data_batch[-1]
            # data_batch = [data.cuda() for data in data_batch[:-1]]
            data_batch = [data.cpu() for data in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, a2c_mask = data_batch
            ner_f1 = cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask)
            time = 0
            rew_partial = 0
            rews, vals, logprobs = [], [], []
            p = 0
            times_mask = sent_length[0] * sent_length[0] * 0.3  # token relation要mask的比例
            # row = [1, 2, 3, 4, 5, 6, 7, 8, 10]  # 初始化节点和节点间的连接关系
            # col = [2, 3, 4, 5, 6, 7, 8, 10, 0]
            row = [0, 1, 2, 3, 4, 5, 6]
            col = [0, 1, 2, 3, 4, 5, 6]
            while time < times_mask:  #
                node_feature = feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                # edge_index = np.zeros(sent_length[0], sent_length[0])
                # for k in range(sent_length[0]):
                #     edge_index[0][k] = k
                # edge_index = torch.Tensor([row, col], dtype=torch.long)
                edge_index = torch.Tensor([row, col])
                edge_index = edge_index.to(dtype=torch.long)
                policy, values = a2c_model(node_feature, edge_index, 4)
                probs = policy.view(-1)  # 转换成一维张量
                # 按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引
                action = torch.distributions.Categorical(
                    logits=probs).sample().detach().item()  # 取样
                action_row = math.floor(action / sent_length[0].item())
                action_col = action % sent_length[0].item()
                is_in = False
                index_p = -1
                for t in range(len(row)):
                    if row[t] == action_row and col[t] == action_col:
                        is_in = True
                        index_p = t
                        break
                if is_in and index_p != -1:
                    del row[index_p]
                    del col[index_p]
                else:
                    row.append(action_row)
                    col.append(action_col)
                for t in range(len(row)):
                    a2c_mask[0][row[t]][col[t]] = 1
                ner_f1_new = cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text,
                                    a2c_mask)
                reward = ner_f1_new - ner_f1
                rew_partial += reward
                # Collect the log-probability of the chosen action
                logprobs.append(policy.view(-1)[action])
                # Collect the value of the chosen action
                vals.append(values)
                # Collect the reward
                rews.append(reward)

                new_state = a2c_mask.clone()
                # After time_to_sample episods we update the loss
                if i % time_to_sample == 0 or time == 10:

                    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
                    vals = torch.stack(vals).flip(dims=(0,)).view(-1)
                    rews = torch.tensor(rews).flip(dims=(0,)).view(-1)

                    # Compute the advantage
                    R = []
                    R_partial = torch.tensor([0.])
                    for j in range(rews.shape[0]):
                        R_partial = rews[j] + gamma * R_partial
                        R.append(R_partial)

                    R = torch.stack(R).view(-1)
                    advantage = R - vals.detach()

                    # Actor loss
                    actor_loss = (-1 * logprobs * advantage)

                    # Critic loss
                    critic_loss = torch.pow(R - vals, 2)

                    # Finally we update the loss
                    optimizer.zero_grad()

                    loss = torch.mean(actor_loss) + \
                           torch.tensor(coeff) * torch.mean(critic_loss)

                    rews, vals, logprobs = [], [], []

                    loss.backward()

                    optimizer.step()
            if p % print_loss == 0:
                print('graph:', p, 'reward:', rew_partial)
            rew_partial = 0
            p += 1



def cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask):
    with torch.no_grad():
        outputs = ner_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
        length = sent_length

        outputs = torch.argmax(outputs, -1)
        ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                            length.cpu().numpy())
        e_f1, e_p, e_r = utils.cal_f1(ent_c, ent_p, ent_r)
        return e_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/weibo-all.json')
    parser.add_argument('--save_path', type=str, default='./model-weibo-origin-base.pt')
    parser.add_argument('--predict_path', type=str, default='./output.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    ner_model = NERModel(config)

    ner_model = ner_model.to(device)

    trainer = Trainer(ner_model)
    trainer.load(config.save_path)
    mode = "train_a2c"
    if mode == "train":
        trainer.predict("Final", test_loader, ori_data[-1])
    else:
        # start using A2C method to train mask leaning
        feature_model = FeatureModel(config)
        units_dim = config.lstm_hid_size
        parser_a2c = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser_a2c.add_argument('--out', default='./temp_edge/', type=str)
        parser_a2c.add_argument(
            "--lr",
            default=0.001,
            help="Learning rate",
            type=float)
        parser_a2c.add_argument(
            "--units",
            default=5,
            help="Number of units in conv layers",
            type=int)
        parser_a2c.add_argument(
            "--batch",
            default=1,
            type=int)
        parser_a2c.add_argument(
            "--gamma",
            default=0.9,
            help="Gamma, discount factor",
            type=float)
        parser_a2c.add_argument(
            "--coeff",
            default=0.1,
            help="Critic loss coefficient",
            type=float)
        parser_a2c.add_argument(
            "--print_loss",
            default=1000,
            help="Steps to take before printing the reward",
            type=int)
        args = parser_a2c.parse_args()

        # torch.manual_seed(1)
        # np.random.seed(2)

        a2c_model = A2CModel(units_dim)
        a2c_model = a2c_model.to(device)

        print(a2c_model)
        print('Model parameters:', sum([w.nelement() for w in a2c_model.parameters()]))

        optimizer = torch.optim.Adam(a2c_model.parameters(), lr=args.lr)

        print('Start training')

        t0 = timeit.default_timer()
        training_a2c(a2c_model, feature_model, train_loader, config.epochs, args.gamma, args.batch, args.coeff, optimizer, args.print_loss)
        ttrain = timeit.default_timer() - t0

        print('Training took:', ttrain, 'seconds')

        torch.save(a2c_model.state_dict(), args.out + '/' + 'model_a2c_mask')
