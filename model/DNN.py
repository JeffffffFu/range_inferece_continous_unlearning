import logging
import joblib
from sympy import false
from torch.utils.data import Subset
from torch.utils.data import TensorDataset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from opacus import PrivacyEngine
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model.ResNet import resnet18, resnet50, resnet18_dp
from model.VGG import vgg11_bn, vgg19_bn, vgg13_bn


class DNN(nn.Module):
    def __init__(self, args=None):
        super(DNN, self).__init__()

        self.logger = logging.getLogger("DNN")
        self.args = args
        self.device = args['device']
        if args['dataset_name']=='tinyimagenet':
            self.imagenet = True
        else:
            self.imagenet = False
        if args['dataset_name']=='cifar100':
            self.num_classes = 100
        elif args['dataset_name']=='celebA':
            self.num_classes = 40
        elif args['dataset_name']=='tinyimagenet':
            self.num_classes = 200
        elif args['dataset_name']=='sst5':
            self.num_classes = 5
        elif args['dataset_name']=='news20':
            self.num_classes = 20
        elif args['dataset_name']=='snli':
            self.num_classes = 3
        elif args['dataset_name']=='mnli':
            self.num_classes = 3
        elif args['dataset_name']=='mrpc':
            self.num_classes = 2
        elif args['dataset_name']=='imdb':
            self.num_classes = 2
        elif args['dataset_name']=='rte':
            self.num_classes = 2
        elif args['dataset_name']=='ag_news':
            self.num_classes = 4
        else:
            self.num_classes = 10
        self.model = self.determine_net(args['net_name'])

    def determine_net(self, net_name, pretrained=False):
        self.logger.debug("determin_net for %s" % net_name)
        self.in_dim = {
            "location": 168,
            "adult": 14,
            "accident": 29,
            "stl10": 96*96*3,
            "cifar10": 32*32*3,
            "cifar100": 32 * 32 * 3,
            "svhn": 32 * 32 * 3,
            "celebA": 128 *128*3,
            "mnist": 28*28*1,
            "fmnist": 28*28*1,
            "cinic10": 32 * 32 * 3,
            "tinyimagenet": 224 * 224 * 3,
            "sst5": 128,
            "news20": 256,
            "snli": 256,
            "mnli": 256,
            "mrpc": 256,
            "imdb": 512,
            "rte": 256,
            "ag_news": 256,
        }
        in_dim = self.in_dim[self.args['dataset_name']]
        out_dim = self.num_classes
        imagenet=self.imagenet
        if net_name == "mlp":
            return MLPTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "logistic":
            return LRTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "simple_cnn":
            return Simple_CNN_Tanh(num_classes=out_dim)
        elif net_name == "simple_cnn_dropout":
            return Simple_CNN_Tanh_dropout(num_classes=out_dim)
        elif net_name == "resnet18":
            return resnet18(num_classes=out_dim)
        elif net_name == "resnet18_dp":
            return resnet18_dp(num_classes=out_dim)
        elif net_name == "resnet50":
            return resnet50(num_classes=out_dim)
        elif net_name == "densenet":
            return models.densenet121(num_classes=out_dim)
        elif net_name == "vgg":
            return vgg11_bn(3, out_dim)
        elif net_name == "CNN_MNIST":
            return CNN_MNIST()
        elif net_name == "DT":
            return DecisionTreeClassifier()
        elif net_name == "RF":
            return RandomForestClassifier()
        elif net_name == "pythia70m":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return Pythia70mModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "pythia70m_dropout":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return Pythia70mModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)
        elif net_name == "roberta":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return RoBERTaModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "roberta_dropout":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return RoBERTaModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)
        elif net_name == "opt13b":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return OPT13BModel(num_classes=out_dim, max_length=max_length)
        elif net_name == "opt13b_dropout":
            # 根据数据集设置max_length
            max_length_map = {
                "sst5": 128,
                "news20": 256,
                "snli": 256,
                "mnli": 256,
                "mrpc": 256,
                "imdb": 512,
                "rte": 256,
                "ag_news": 256,
            }
            max_length = max_length_map.get(self.args['dataset_name'], 128)
            return OPT13BModel_dropout(num_classes=out_dim, max_length=max_length, dropout_rate=0.95)

        else:
            raise Exception("invalid net name")

    def train_model(self, train_loader, test_loader, save_name=None):
        self.model = self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'],weight_decay=1e-4)
        if self.args['optim'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=1e-4)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(secure_mode=False)

            self.model, optimizer, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=0.5,
                max_grad_norm=1.0,
            )
        criterion = nn.CrossEntropyLoss()
        run_result = []

        self.model.train()
        test_acc=0.
        for epoch in range(self.args['num_epochs']):
            losses = []

            for batch in train_loader:
                # 处理不同类型的数据
                if isinstance(batch, dict):
                    # 文本数据（SST5等）
                    data = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    target = batch['labels'].to(self.device)
                else:
                    # 图像数据
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                    
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if self.args['max_norm'] > 0:  #for certified removal
                    param_norm = nn.utils.parameters_to_vector(self.model.parameters()).norm()
                    if param_norm > self.args['max_norm']:
                        scale_factor = self.args['max_norm'] / param_norm
                        for param in self.model.parameters():
                            param.data *= scale_factor

            if self.args['is_dp_defense']:
                epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.2f}, δ = 1e-5)"
                )

            train_acc = self.test_model_acc(train_loader)
            test_acc = self.test_model_acc(test_loader)
            print(f' epoch:{epoch} | train acc:{round(train_acc, 4)} | test acc: {round(test_acc, 4)}')

        # # self.logger.debug('epoch %s: train acc %s | test acc %s | ovf %s' % (epoch, train_acc, test_acc, train_acc - test_acc))
        # run_result.append([epoch, np.mean(losses), train_acc, test_acc])
        #


    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))

    def predict_proba2(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            feature = test_case[0][0]
            feature = torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()

    def predict_proba(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            if isinstance(test_case, dict):
                # 文本数据
                feature = {k: v.unsqueeze(0).to(self.device) for k, v in test_case.items() if k != 'labels'}
            else:
                # 图像数据
                feature = torch.unsqueeze(test_case.to(torch.float32), 0).to(self.device)
            
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            return posterior.detach().cpu().numpy()
    # def predict_proba(self, test_case):
    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     with torch.no_grad():
    #         logits = self.model(test_case)
    #         posterior = F.softmax(logits, dim=1)
    #         return posterior.detach().cpu().numpy()

    def test_model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for batch in test_loader:
                # 处理不同类型的数据
                if isinstance(batch, dict):
                    # 文本数据（SST5等）
                    data = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    target = batch['labels'].to(self.device)
                else:
                    # 图像数据
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            return correct / len(test_loader.dataset)

    def logits(self,target_sample):
        if isinstance(target_sample, Subset):
            with torch.no_grad():
                feature=target_sample[0][0]
                label=target_sample[0][1]
                feature =torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
                label=torch.unsqueeze(torch.tensor(label, dtype=torch.long),0).to(self.device)
                logits = self.model(feature).to(self.device)
                logits=logits[0].detach().cpu().numpy()
                probs=np.exp(logits)/np.sum(np.exp(logits))
               # confidence=probs[label]
                confidence=np.max(probs)
                # confidence = np.clip(confidence, 1e-10, 1 - 1e-10)
                # confidence=np.log(confidence/(1-confidence))
        else:
            ValueError("This is not a Subset.")
        return confidence

    def forward_propagation(self, target_sample):
        self.model.eval()
        self.model = self.model.to(self.device)
        return self.model(target_sample)

    def forward(self,x):
        x=self.model(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d( in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x



class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1, 16, 8, 2, padding=2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Conv2d(16, 32, 4, 2),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 1),
                                      nn.Flatten(),
                                      nn.Linear(32 * 4 * 4, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 10))
    def forward(self,x):
        if x.dim() == 2:  # 如果输入形状是 (batch_size, 10)
            x = x.unsqueeze(1).unsqueeze(3)  # 调整为 (batch_size, 1, 10, 1)
        x=self.conv(x)
        return x

def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class Simple_CNN_Tanh(nn.Module):
    def __init__(self,num_classes=10, in_channels=3, input_norm=None,**kwargs):
        super(Simple_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.num_classes=num_classes

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh
       # act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, self.num_classes))
        else:
            self.classifier = nn.Linear(c * 4 * 4, self.num_classes)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Simple_CNN_Tanh_dropout(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_norm=None, drop_p=0.95, **kwargs):
        super(Simple_CNN_Tanh_dropout, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_p = drop_p

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        c = self.in_channels

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.drop_p > 0:
                    layers += [nn.Dropout2d(self.drop_p)]  # 对2D特征图的空间Dropout
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(
                nn.Linear(c * 4 * 4, hidden),
                act(),
                nn.Dropout(self.drop_p),  # 全连接层后添加Dropout
                nn.Linear(hidden, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.drop_p),  # 输入层Dropout
                nn.Linear(c * 4 * 4, self.num_classes)
            )
    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class MLPTorchNet(nn.Module):
    def __init__(self, in_dim=168, out_dim=9):
        super(MLPTorchNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # temperature = 4
        # x /= temperature
        # return F.log_softmax(x, dim=1)
        return x


class LRTorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Pythia70mModel(nn.Module):
    """Pythia-70m模型包装器，用于文本分类任务"""
    def __init__(self, num_classes=5, max_length=128, model_name="EleutherAI/pythia-70m"):
        super(Pythia70mModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: 包含input_ids, attention_mask的字典或张量
        """
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            # 如果输入是张量，假设是input_ids
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class Pythia70mModel_dropout(nn.Module):
    def __init__(self, num_classes=5, max_length=128, model_name="EleutherAI/pythia-70m", dropout_rate=0.95):
        super(Pythia70mModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):

        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        """动态设置dropout率"""
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        self.model.config.hidden_dropout = dropout_rate
        self.model.config.attention_dropout = dropout_rate
        self.model.config.classifier_dropout = dropout_rate


class RoBERTaModel(nn.Module):
    """RoBERTa模型包装器，用于文本分类任务"""
    def __init__(self, num_classes=5, max_length=128, model_name="roberta-base"):
        super(RoBERTaModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: 包含input_ids, attention_mask的字典或张量
        """
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            # 如果输入是张量，假设是input_ids
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class RoBERTaModel_dropout(nn.Module):
    """RoBERTa模型包装器（带dropout），用于文本分类任务"""
    def __init__(self, num_classes=5, max_length=128, model_name="roberta-base", dropout_rate=0.95):
        super(RoBERTaModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        """动态设置dropout率"""
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        if hasattr(self.model.config, 'hidden_dropout_prob'):
            self.model.config.hidden_dropout_prob = dropout_rate
        if hasattr(self.model.config, 'attention_probs_dropout_prob'):
            self.model.config.attention_probs_dropout_prob = dropout_rate
        if hasattr(self.model.config, 'classifier_dropout'):
            self.model.config.classifier_dropout = dropout_rate


class OPT13BModel(nn.Module):
    """OPT-1.3B模型包装器，用于文本分类任务"""
    def __init__(self, num_classes=5, max_length=128, model_name="facebook/opt-1.3b"):
        super(OPT13BModel, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        """
        前向传播
        Args:
            batch: 包含input_ids, attention_mask的字典或张量
        """
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            # 如果输入是张量，假设是input_ids
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name


class OPT13BModel_dropout(nn.Module):
    """OPT-1.3B模型包装器（带dropout），用于文本分类任务"""
    def __init__(self, num_classes=5, max_length=128, model_name="facebook/opt-1.3b", dropout_rate=0.95):
        super(OPT13BModel_dropout, self).__init__()
        self.max_length = max_length
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            hidden_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            classifier_dropout=dropout_rate
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
        else:
            input_ids = batch
            attention_mask = torch.ones_like(input_ids)
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = self.dropout(outputs.logits)
        
        return logits
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_name(self):
        return self.model_name
    
    def set_dropout_rate(self, dropout_rate):
        """动态设置dropout率"""
        self.dropout_rate = dropout_rate
        self.dropout.p = dropout_rate
        self.model.config.hidden_dropout = dropout_rate
        self.model.config.attention_dropout = dropout_rate
        self.model.config.classifier_dropout = dropout_rate
    
