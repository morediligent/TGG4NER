import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
import math
import random
import pickle

import config
import data_loader
import utils
from ner_model import Model as FeatureModel
from a2c_model import A2CModel
from ner_model_origin import Model as NERModel

import timeit
import os
import os.path as osp

# 获取当前脚本的绝对路径
script_path = osp.abspath(osp.dirname(__file__))

# 切换到脚本所在的目录
os.chdir(script_path)

# 打印当前工作目录确认是否已经改变
print("当前工作目录已更改为:", os.getcwd())

class Trainer(object):
    def __init__(self, ner_model, a2c_model, feature_model, args):
        self.ner_model = ner_model
        self.a2c_model = a2c_model
        self.optimizer = torch.optim.Adam(self.a2c_model.parameters(), lr=args.lr)
        self.feature_model = feature_model

    def predict(self, epoch, data_loader, data, mode, data_type):
        self.ner_model.eval()

        pred_result = []
        label_result = []

        result = []
        result_a2c_mask = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        i = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch1 = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, a2c_mask = data_batch1
                if mode == "ner_a2c" or mode == "ner_a2c_train":
                    a2c_mask, ner_f1, ner_f1_new = self.predict_a2c(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask, entity_text, i)
                    result_a2c_mask.append(a2c_mask)
                    
                outputs = self.ner_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())
                
                # e_f1, e_p, e_r = utils.cal_f1(ent_c, ent_p, ent_r)
                # logger.info('origin f1 {}, a2cmask f1 {}, a2cmask e_f1 {}'.format(ner_f1, ner_f1_new, e_f1))
                # for ent_list, sentence in zip(decode_entities, sentence_batch):
                #     sentence = sentence["sentence"]
                #     instance = {"sentence": sentence, "entity": []}
                #     for ent in ent_list:
                #         instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                #                                    "type": config.vocab.id_to_label(ent[1])})
                #     result.append(instance)

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

        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.cpu().numpy(),
                                                             pred_result.cpu().numpy(),
                                                             average=None)))

        table = pt.PrettyTable(["{} {}".format(config.dataset, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        if mode == "ner_a2c" or mode == "ner_a2c_train":
            with open('./a2c_mask/'+config.dataset+'_'+data_type+'_a2cmask_EM0.pkl', 'wb') as file:
                pickle.dump(result_a2c_mask, file)

        return e_f1, e_p, e_r

    def save(self, path):
        torch.save(self.ner_model.state_dict(), path)

    def load(self, path):
        self.ner_model.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))
        # self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # cpu load

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
                for j in range(sent_length[0]):
                    if j != k:
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
                
                    # if reward != 0:
                    #     print("reward",reward)
                    if reward > 0:
                        action = action_list.item()
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

    def training_a2c(self, data_loader, args):  
        self.a2c_model.train() 
        for i, data_batch in enumerate(data_loader):
            entity_text = data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, _, grid_mask2d, pieces2word, dist_inputs, sent_length, a2c_mask = data_batch
            with torch.no_grad(): # 80M
                node_feature = self.feature_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
                ner_f1 = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask) 
        
            time = 0
            rew_partial = 0
            rews, vals, logprobs = [], [], []
            temp_ner_f1 = ner_f1
            for k in range(sent_length[0]):       
                row = []
                col = []
                a2c_mask[0][k][k] = 1
                for j in range(sent_length[0]):
                    if j != k:
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
                    policy, values = self.a2c_model(node_feature, edge_index, k)  
                    probs = policy.view(-1)  # 转换成一维张量
                    # 按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引  
                    action_list = torch.distributions.Categorical(
                        logits=probs).sample(sample_shape=(args.time_to_update,)).detach()   # 取样规则可更改，按概率来  sample_shape的值？num_samples
                    for action in action_list:
                        action = action.item()
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
                                
                    with torch.no_grad(): 
                        ner_f1_new = self.cal_ner_f1(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text,
                                        a2c_mask)
                    
                    
                    reward = ner_f1_new - temp_ner_f1  # ner_f1用origin还是pre
                    temp_ner_f1 = ner_f1_new
                    rew_partial += reward
                    # Collect the log-probability of the chosen action
                    logprobs.append(policy.view(-1)[action])
                    # Collect the value of the chosen action
                    vals.append(values)
                    # Collect the reward
                    rews.append(reward)
                time = time + 1
                # After time_to_sample episods we update the loss
                if time % args.time_to_update == 0 or time == sent_length[0]:

                    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
                    vals = torch.stack(vals).flip(dims=(0,)).view(-1)
                    rews = torch.tensor(rews).flip(dims=(0,)).view(-1)

                    # Compute the advantage
                    R = []
                    R_partial = torch.tensor([0.])
                    for j in range(rews.shape[0]):
                        R_partial = rews[j] + args.gamma * R_partial
                        R.append(R_partial)

                    R = torch.stack(R).view(-1).to(device)
                    advantage = R - vals.detach()

                    actor_loss = (-1 * logprobs * advantage)
                    critic_loss = torch.pow(R - vals, 2)   
                                     

                    loss = torch.mean(actor_loss) + \
                            torch.tensor(args.coeff) * torch.mean(critic_loss)

                    rews, vals, logprobs = [], [], []

                    self.optimizer.zero_grad()
                    loss.backward() 
                    self.optimizer.step() 
                    
            logger.info("sent_index:{}, total reward:{}".format(i, round(rew_partial, 2)))
            rew_partial = 0

    def cal_ner_f1(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, entity_text, a2c_mask):
        outputs = self.ner_model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, a2c_mask)
        length = sent_length

        outputs = torch.argmax(outputs, -1)
        ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                            length.cpu().numpy())
        e_f1, e_p, e_r = utils.cal_f1(ent_c, ent_p, ent_r)
        return e_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/resume-zh.json')
    parser.add_argument('--save_path', type=str, default='model.pt')
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

    config.batch_size = 1

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=0,  # shuffle=i == 0  
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    ner_model = NERModel(config)
    ner_model = ner_model.to(device)  # 1.5G
    units_dim = config.bert_hid_size  #  lstm_hid_size？
    a2c_model = A2CModel(units_dim)
    a2c_model = a2c_model.to(device)  # 8M
    feature_model = FeatureModel(config)
    feature_model.to(device)
    # print(a2c_model)
    # print('Model parameters:', sum([w.nelement() for w in a2c_model.parameters()]))

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
    parser_a2c.add_argument(
        "--sample_num",
        default=4,
        help="the number of sample",
        type=int)
    parser_a2c.add_argument(
        "--mask_ratio",
        default=0.4,
        help="the ration of mask",
        type=int)
    parser_a2c.add_argument(
        "--time_to_update",
        default=4,
        help="the time of update",
        type=int)
    parser_a2c.add_argument(
        "--epochs",
        default=2,
        help="the number of epochs",
        type=int)
    parser_a2c.add_argument(
        "--thepoch",
        default=0,
        help="the order of epochs",
        type=str)
    parser_a2c.add_argument('--mode', type=str, default='ner_a2c')  #ner_a2c_train   ner    ner_a2c
    args = parser_a2c.parse_args()
    logger.info("{}".format(args))
    trainer = Trainer(ner_model, a2c_model, feature_model, args)
    trainer.ner_model.load_state_dict(torch.load('/home/wuwj/pycode/W2NER-main/model_pt/'+config.dataset+'_origin_'+config.save_path, map_location=torch.device('cuda:0')))
    trainer.feature_model.load_state_dict(torch.load('/home/wuwj/pycode/W2NER-main/model_pt/'+config.dataset+'_origin_'+config.save_path, map_location=torch.device('cuda:0')))
    # trainer.a2c_model.load_state_dict(torch.load('/home/wuwj/pycode/v16_drl_a2c_mask/mask_model/conll03_a2cmask_1th.pt', map_location=torch.device('cuda:0')))
    if args.mode == "ner":
        trainer.predict("Final", test_loader, ori_data[-1], args.mode, "train")
    elif args.mode == "ner_a2c_train":
        print('Start training')
        best_f1, best_p, best_r = 0.0, 0.0, 0.0
        for j in range(args.epochs):
            logger.info("epoch:{}".format(j))
            t0 = timeit.default_timer()
            trainer.training_a2c(train_loader, args)
            ttrain = timeit.default_timer() - t0
            print('Training took:', ttrain, 'seconds')
            thepoch = str(j)
            torch.save(trainer.a2c_model.state_dict(), './mask_model/'+config.dataset+'_a2cmask_'+thepoch+'th.pt')
            print('Start predicting')
            t0 = timeit.default_timer()
            e_f1, e_p, e_r = trainer.predict("Final", test_loader, ori_data[-1],"ner_a2c", "test")
            if e_f1 > best_f1:
                best_f1 = e_f1
                best_p = e_p
                best_r = e_r
                torch.save(trainer.a2c_model.state_dict(), './mask_model/'+config.dataset+'_best_a2cmask_'+thepoch+'th.pt')
        table = pt.PrettyTable(["{} {}".format(config.dataset, j), 'F1', "Precision", "Recall"])
        table.add_row(["best Entity"] + ["{:3.4f}".format(x) for x in [best_f1, best_p, best_r]])
        logger.info("\n{}".format(table))

    else:
        print('Start predicting')
        t0 = timeit.default_timer()
        trainer.a2c_model.load_state_dict(torch.load('/home/wuwj/pycode/v16_drl_a2c_mask/mask_model/'+config.dataset+'_a2cmask_0th.pt', map_location=torch.device('cuda:0'))) 
        e_f1, e_p, e_r = trainer.predict("Final", train_loader, ori_data[-3],"ner_a2c", "train")

        e_f1, e_p, e_r = trainer.predict("Final", dev_loader, ori_data[-2],"ner_a2c", "dev")

        e_f1, e_p, e_r = trainer.predict("Final", test_loader, ori_data[-1],"ner_a2c", "test")

