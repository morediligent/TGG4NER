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
import pickle

import config
import data_loader as data_loader
import utils
from ner_model_origin import Model 
from a2c_model import A2CModel
from ner_model import Model as FeatureModel

import os
import os.path as osp

# 获取当前脚本的绝对路径
script_path = osp.abspath(osp.dirname(__file__))

# 切换到脚本所在的目录
os.chdir(script_path)

# 打印当前工作目录确认是否已经改变
print("当前工作目录已更改为:", os.getcwd())


class Trainer(object):
    def __init__(self, model, a2c_model, feature_model):
        self.model = model
        self.a2c_model = a2c_model
        self.feature_model = feature_model
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
        # self.scheduler = transformers.WarmupLinearSchedule(self.optimizer,
        #                                                     warmup_steps=config.warm_factor * updates_total,
        #                                                     t_total =updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            temp_batch = data_batch
            entity_text= data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, batch_a2c_mask = data_batch

            # start to get mask with batch_size
            a2c_mask = []
            max_tok = torch.max(sent_length)
            for batch_index in range(len(sent_length)):
                temp_length = sent_length[batch_index]
                _bert_inputs = bert_inputs[batch_index][:temp_length+2].unsqueeze(0)
                _grid_mask2d = grid_mask2d[batch_index][:temp_length, :temp_length].unsqueeze(0)
                _dist_inputs = dist_inputs[batch_index][:temp_length, :temp_length].unsqueeze(0)
                _pieces2word = pieces2word[batch_index][:temp_length, :temp_length+2].unsqueeze(0)
                _batch_a2c_mask = batch_a2c_mask[batch_index][:temp_length, :temp_length].unsqueeze(0)

                temp_entity_text = []
                temp_entity_text.append(entity_text[batch_index])
                _a2c_mask, ner_f1, ner_f1_new = self.predict4a2c(_bert_inputs, _grid_mask2d, _dist_inputs, _pieces2word, temp_length.unsqueeze(0), _batch_a2c_mask, temp_entity_text, i)
                _a2c_mask = _a2c_mask.squeeze(0)
                a2c_mask.append(_a2c_mask)
            
            a2c_mask_mat = torch.zeros((len(sent_length), max_tok, max_tok), dtype=torch.bool)
            a2c_mask = self.fill(a2c_mask, a2c_mask_mat)
            a2c_mask = a2c_mask.to(device)
            # end to get mask with batch_size
           
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, batch_a2c_mask = data_batch

                if i >=72:
                    print(i)

                # start to get mask with batch_size
                a2c_mask = []
                max_tok = torch.max(sent_length)
                for batch_index in range(len(sent_length)):
                    temp_length = sent_length[batch_index]
                    _bert_inputs = bert_inputs[batch_index][:temp_length+2].unsqueeze(0)
                    _grid_mask2d = grid_mask2d[batch_index][:temp_length, :temp_length].unsqueeze(0)
                    _dist_inputs = dist_inputs[batch_index][:temp_length, :temp_length].unsqueeze(0)
                    _pieces2word = pieces2word[batch_index][:temp_length, :temp_length+2].unsqueeze(0)
                    _batch_a2c_mask = batch_a2c_mask[batch_index][:temp_length, :temp_length].unsqueeze(0)

                    temp_entity_text = []
                    temp_entity_text.append(entity_text[batch_index])
                    if is_test:
                        _a2c_mask, ner_f1, ner_f1_new = self.predict_a2c(_bert_inputs, _grid_mask2d, _dist_inputs, _pieces2word, temp_length.unsqueeze(0), _batch_a2c_mask, temp_entity_text, i)
                    else:
                        _a2c_mask, ner_f1, ner_f1_new = self.predict_a2c(_bert_inputs, _grid_mask2d, _dist_inputs, _pieces2word, temp_length.unsqueeze(0), _batch_a2c_mask, temp_entity_text, i)
                    _a2c_mask = _a2c_mask.squeeze(0)
                    a2c_mask.append(_a2c_mask)
                
                a2c_mask_mat = torch.zeros((len(sent_length), max_tok, max_tok), dtype=torch.bool)
                a2c_mask = self.fill(a2c_mask, a2c_mask_mat)
                a2c_mask = a2c_mask.to(device)
                # end to get mask with batch_size

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))
        return e_f1

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
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, batch_a2c_mask = data_batch
                # start to get mask with batch_size
                a2c_mask = []
                max_tok = torch.max(sent_length)
                for batch_index in range(len(sent_length)):
                    temp_length = sent_length[batch_index]
                    _bert_inputs = bert_inputs[batch_index][:temp_length+2].unsqueeze(0)
                    _grid_mask2d = grid_mask2d[batch_index][:temp_length, :temp_length].unsqueeze(0)
                    _dist_inputs = dist_inputs[batch_index][:temp_length, :temp_length].unsqueeze(0)
                    _pieces2word = pieces2word[batch_index][:temp_length, :temp_length+2].unsqueeze(0)
                    _batch_a2c_mask = batch_a2c_mask[batch_index][:temp_length, :temp_length].unsqueeze(0)

                    temp_entity_text = []
                    temp_entity_text.append(entity_text[batch_index])
                    _a2c_mask, ner_f1, ner_f1_new = self.predict_a2c(_bert_inputs, _grid_mask2d, _dist_inputs, _pieces2word, temp_length.unsqueeze(0), _batch_a2c_mask, temp_entity_text, i)
                    _a2c_mask = _a2c_mask.squeeze(0)
                    a2c_mask.append(_a2c_mask)
                
                a2c_mask_mat = torch.zeros((len(sent_length), max_tok, max_tok), dtype=torch.bool)
                a2c_mask = self.fill(a2c_mask, a2c_mask_mat)
                a2c_mask = a2c_mask.to(device)
                # end to get mask with batch_size
                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

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

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open("./ner_model/"+config.dataset+"_EM0_"+config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return e_f1

    def predict_a2c_backup(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask, entity_text, sent_index):
        self.a2c_model.eval()
        with torch.no_grad():
            node_feature = self.feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
            ner_f1 = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask)
            time = 0
            rew_partial = 0
            rews, vals, logprobs = [], [], []
            temp_ner_f1 = ner_f1

            for k in range(sent_length[0]):       
                row = []
                col = []
                temp_mask = a2c_mask.clone()
                temp_mask[0][k][k] = 1
                ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, temp_mask)
                if ner_f1_new - ner_f1 >= 0:
                    a2c_mask[0][k][k] = 1
                for j in range(sent_length[0]):
                        row.append(k)
                        col.append(j)
                num_samples = math.ceil(sent_length[0] * args.mask_ratio)  # token relation要mask的比例
                times_update = math.ceil(num_samples / args.time_to_update)
                for index_update in range(times_update): 
                    edge_index = torch.Tensor([row, col])
                    if edge_index.shape[0] != 2:
                        print("edge_index shape0 is not 2")
                    edge_index = edge_index.to(dtype=torch.long)
                    edge_index = edge_index.to(device)
                    policy = self.a2c_model(node_feature, edge_index, k)  
                    probs = policy.view(-1)  # 转换成一维张量
                    # 按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引  
                    # action_list = torch.distributions.Categorical(
                    #     logits=probs).sample(sample_shape=(args.time_to_update,)).detach()   # 取样规则可更改，按概率来  sample_shape的值？num_samples
                    action_list = torch.distributions.Categorical(logits=probs).sample(sample_shape=()).detach()
                    temp_mask = a2c_mask.clone()
                    action = action_list.item()
                    if temp_mask[0][k][action] == 1 :
                        temp_mask[0][k][action] = 0
                    else:
                        temp_mask[0][k][action] = 1
                                
                    ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text,
                                        temp_mask)
                    
                    reward = ner_f1_new - temp_ner_f1  # ner_f1用origin还是pre           
                
                    if reward > 0:
                        if a2c_mask[0][k][action] == 1 :
                            a2c_mask[0][k][action] = 0
                            row.append(k)
                            col.append(action)
                        else:
                            a2c_mask[0][k][action] = 1
                            index_p = col.index(action)
                            if index_p !=-1:
                                del row[index_p]
                                del col[index_p]
                        temp_ner_f1 = ner_f1_new
                        rew_partial += reward
                time = time + 1
                
        logger.info("sent_index:{}, total reward:{}".format(sent_index, round(rew_partial, 2)))
        return a2c_mask, ner_f1, ner_f1_new
    
    def predict_a2c(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask, entity_text, sent_index):
        self.a2c_model.eval()
        with torch.no_grad():
            node_feature = self.feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
            ner_f1 = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask)
            time = 0
            rew_partial = 0
            rews, vals, logprobs = [], [], []
            temp_ner_f1 = ner_f1

            for k in range(sent_length[0]):       
                row = []
                col = []
                temp_mask = a2c_mask.clone()
                temp_mask[0][k][k] = 1
                ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, temp_mask)
                if ner_f1_new - ner_f1 >= 0:
                    a2c_mask[0][k][k] = 1
                row = torch.full((sent_length.item(),), k, dtype=torch.long)
                col = torch.arange(sent_length.item(), dtype=torch.long)
                num_samples = math.ceil(sent_length[0] * args.mask_ratio)  # token relation要mask的比例
                times_update = math.ceil(num_samples / args.time_to_update)
                edge_index = torch.stack([row, col], dim=0)
                if edge_index.shape[0] != 2:
                    print("edge_index shape0 is not 2")
                edge_index = edge_index.to(device)
                policy = self.a2c_model(node_feature, edge_index, k)  
                probs = policy.view(-1)  # 转换成一维张量
                action_list = torch.distributions.Categorical(logits=probs).sample(sample_shape=([times_update])).detach()
                temp_mask = a2c_mask.clone()
                for action in action_list:
                    action = action.item()
                    if temp_mask[0][k][action] == 0 :
                        temp_mask[0][k][action] = 1
                                
                    ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text,
                                        temp_mask)
                    
                    reward = ner_f1_new - temp_ner_f1  # ner_f1用origin还是pre           
                
                    if reward > 0:
                        if a2c_mask[0][k][action] == 0 :
                            a2c_mask[0][k][action] = 1
                        temp_ner_f1 = ner_f1_new
                        rew_partial += reward
                time = time + 1
                
        logger.info("sent_index:{}, total reward:{}".format(sent_index, round(rew_partial, 2)))
        return a2c_mask, ner_f1, ner_f1_new
    
    def predict4a2c_backup(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask, entity_text, sent_index):
        self.a2c_model.eval()
        with torch.no_grad():
            node_feature = self.feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
            ner_f1 = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask)
            time = 0
            rew_partial = 0
            rews, vals, logprobs = [], [], []
            temp_ner_f1 = ner_f1

            for k in range(sent_length.item()):       
                row = []
                col = []
                a2c_mask[0][k][k] = 1
                for j in range(sent_length.item()):
                    row.append(k)
                    col.append(j)
                num_samples = math.ceil(sent_length[0] * args.mask_ratio)  # token relation要mask的比例
                times_update = math.ceil(num_samples / args.time_to_update)
                for index_update in range(times_update): 
                    edge_index = torch.Tensor([row, col])
                    edge_index = edge_index.to(dtype=torch.long)
                    edge_index = edge_index.to(device)
                    policy = self.a2c_model(node_feature, edge_index, k)  
                    probs = policy.view(-1)  # 转换成一维张量
                    action_list = torch.distributions.Categorical(logits=probs).sample(sample_shape=()).detach()
                    action = action_list.item()
                    if a2c_mask[0][k][action] == 1 : # 1没有联系，0有联系
                        a2c_mask[0][k][action] = 0
                        row.append(k)
                        col.append(action)
                    else: # 测试下
                        a2c_mask[0][k][action] = 1
                        index_p = col.index(action)
                        if index_p !=-1:
                            del row[index_p]
                            del col[index_p]
                                
                    ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text,
                                        a2c_mask)
                    reward = ner_f1_new - temp_ner_f1  # ner_f1用origin还是pre
                    temp_ner_f1 = ner_f1_new         
                    rew_partial += reward
                time = time + 1
                
        logger.info("sent_index:{}, total reward:{}".format(sent_index, round(rew_partial, 2)))
        return a2c_mask, ner_f1, ner_f1_new
    
    def predict4a2c(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask, entity_text, sent_index):
        self.a2c_model.eval()
        with torch.no_grad():
            node_feature = self.feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)

            for k in range(sent_length.item()):       
                row = []
                col = []
                a2c_mask[0][k][k] = 1
                # 创建行索引和列索引
                row = torch.full((sent_length.item(),), k, dtype=torch.long)
                col = torch.arange(sent_length.item(), dtype=torch.long)
                num_samples = math.ceil(sent_length[0] * args.mask_ratio)  # token relation要mask的比例
                times_update = math.ceil(num_samples / args.time_to_update)
                edge_index = torch.stack([row, col], dim=0)
                edge_index = edge_index.to(dtype=torch.long)
                edge_index = edge_index.to(device)
                policy = self.a2c_model(node_feature, edge_index, k)  
                probs = policy.view(-1)  # 转换成一维张量
                action_list = torch.distributions.Categorical(logits=probs).sample(sample_shape=([times_update])).detach()
                for action in action_list:
                    action = action.item()
                    if a2c_mask[0][k][action] == 0 : # 1没有联系，0有联系
                        a2c_mask[0][k][action] = 1
                
        return a2c_mask, 0.0, 0.0
    

    def cal_ner_f1(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask):
        outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
        length = sent_length

        outputs = torch.argmax(outputs, -1)
        ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                            length.cpu().numpy())
        e_f1, e_p, e_r = utils.cal_f1(ent_c, ent_p, ent_r)
        return e_f1
    
    def fill(self, data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ace05.json')
    parser.add_argument('--save_path', type=str, default='model.pt')
    parser.add_argument('--predict_path', type=str, default='output.json')
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

    args.mask_ratio = 0.4
    args.time_to_update = 4

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

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

    logger.info("Building Model")
    model = Model(config)

    model = model.to(device)
    a2c_model = A2CModel(config.bert_hid_size)
    a2c_model = a2c_model.to(device)
    feature_model = FeatureModel(config)
    feature_model = feature_model.to(device)

    trainer = Trainer(model, a2c_model, feature_model)
    trainer.model.load_state_dict(torch.load('/home/wuwj/pycode/W2NER-main/model_pt/'+config.dataset+'_origin_'+config.save_path, map_location=torch.device('cuda:0')))
    trainer.a2c_model.load_state_dict(torch.load('/home/wuwj/pycode/v16_drl_a2c_mask/mask_model/'+config.dataset+'_a2cmask_0th.pt', map_location=torch.device('cuda:0')))
    trainer.feature_model.load_state_dict(torch.load('/home/wuwj/pycode/W2NER-main/model_pt/'+config.dataset+'_origin_'+config.save_path, map_location=torch.device('cuda:0')))
    mode = "train"
    if mode == "train":
        #  f1 = trainer.eval(0, dev_loader)
        best_f1 = 0
        best_test_f1 = 0
        for i in range(config.epochs):
            logger.info("Epoch: {}".format(i))
            trainer.train(i, train_loader)
            f1 = trainer.eval(i, dev_loader)
            test_f1 = trainer.eval(i, test_loader, is_test=True)
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                trainer.save('./ner_model/'+config.dataset+'_EM0_'+config.save_path)
        logger.info("Best DEV F1: {:3.4f}".format(best_f1))
        logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
        trainer.load('./ner_model/'+config.dataset+'_EM0_'+config.save_path)
        trainer.predict("Final", test_loader, ori_data[-1])
    elif mode =="predict":
        logger.info("Start Predicting")
        trainer.predict("Final", test_loader, ori_data[-1])
