import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0, metadata=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
        self.metadata = metadata if isinstance(metadata, dict) else None

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            checkpoint_path = self.save_checkpoint(val_loss, val_loss2, model, path)
            return checkpoint_path
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            checkpoint_path = self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0
            return checkpoint_path

    def save_checkpoint(self, val_loss, val_loss2, model, path):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 创建checkpoint子文件夹
        ckpt_dir_name = f"{self.dataset}_checkpoint_{timestamp}"
        ckpt_dir = os.path.join(path, ckpt_dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 保存模型到model.pth
        model_path = os.path.join(ckpt_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # 保存运行参数配置到config.json
        if self.metadata is not None:
            try:
                import json
                cfg_path = os.path.join(ckpt_dir, "config.json")
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        
        # 返回保存的checkpoint目录路径
        return ckpt_dir

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        # 保存原始运行配置，便于直接落盘
        try:
            self.run_config = dict(config)
        except Exception:
            self.run_config = None

        # 生成安全的检查点文件名前缀，避免将路径嵌入文件名导致多级目录
        base_name = os.path.splitext(os.path.basename(str(self.data_path)))[0]
        self.checkpoint_stem = ''.join([c if (str(c).isalnum() or c in ('-', '_')) else '_' for c in base_name])
        
        # 存储最新保存的checkpoint目录路径
        self.latest_checkpoint_dir = None

        # 直接使用传入的 data_path（对于 Custom 数据集应为Parquet文件路径）
        data_source = self.data_path
        self.train_loader = get_loader_segment(self.index, data_source, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index, data_source, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, data_source, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, data_source, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())
            loss_2.append((prior_loss + series_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        # 直接保存原始运行配置
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.checkpoint_stem, metadata=self.run_config)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            checkpoint_dir = early_stopping(vali_loss1, vali_loss2, self.model, path)
            if checkpoint_dir is not None:
                self.latest_checkpoint_dir = checkpoint_dir
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        # 优先使用训练时保存的最新checkpoint目录
        if self.latest_checkpoint_dir is not None and os.path.isdir(self.latest_checkpoint_dir):
            ckpt_dir = self.latest_checkpoint_dir
        
        else:
            raise FileNotFoundError(f"No checkpoint directory found in {self.latest_checkpoint_dir}")
        
        # 加载模型文件
        model_path = os.path.join(ckpt_dir, "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path))
        print(f"Loading checkpoint from: {model_path}")
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        # 保存测试结果到result.json
        try:
            import json
            result_payload = {
                "threshold": float(thresh),
                "metrics": {k: float(v) for k, v in scores_simple.items()},
                "summary": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f_score": float(f_score)
                }
            }
            result_path = os.path.join(ckpt_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result_payload, f, ensure_ascii=False, indent=2)
            print(f"Test results saved to: {result_path}")
        except Exception as e:
            print(f"Failed to save test results: {e}")
    
        return accuracy, precision, recall, f_score

    def _export_run_config(self):
        # 保留以兼容调用，但优先使用 self.run_config
        return self.run_config
