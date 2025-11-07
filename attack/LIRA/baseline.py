import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from parameter_parser import parameter_parser
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
import os
from unlearning.retrain3 import sample_target_samples
from model.DNN import DNN
import pandas as pd


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list=[64, 32], device=torch.device('cpu')):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.device = torch.device('cpu')
        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def train_epochs(self, train_x, train_y, val_x, val_y, epochs=200, max_patient=20):
        pbar = tqdm(range(epochs), leave=True, desc=f"Attack Training")
        if val_x is None:
           val_x = train_x
           val_y = train_y
        opt_loss = 1e10
        patient = max_patient
        for i in pbar:
            self.train_one_epoch(train_x, train_y)
            train_loss, train_acc = self.loss_acc(train_x, train_y)
            val_loss, val_acc = self.loss_acc(val_x, val_y)

            pbar.set_postfix({'Loss': train_loss,
                              'Acc': train_acc,
                              'Val Loss': val_loss,
                              'Val Acc': val_acc})
            if opt_loss / 1.001 > val_loss:
                opt_loss = val_loss
                patient = max_patient
            else:
                patient = patient - 1

            if patient == 0:
                pbar.close()
                break

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)
        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def acc(self, X_target, y_target):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        acc = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        return acc

    def pred(self, X_target):
        '''Predict posteriors'''
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        return outputs_target.detach().numpy()


    def all_metrics(self, X_target, y_target):

        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy()[:, 1])
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))

        tpr_05fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.05)
        tpr_10fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.10)
        tpr_001fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.01)
        return [acc_target, prec_target, recall_target, auc_target, f1_target,
                tpr_001fpr, tpr_05fpr, tpr_10fpr]


def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = max(tpr[fpr==fpr[ind]])
    return tpr_res


def attack_feature_base(P_in, P_out, method="DC"):
    assert method in ["DC", "SC", "DD", "SD", "ED"]

    if method == "DC":
        return np.concatenate([P_in, P_out], axis=1)
    elif method == "SC":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return np.concatenate([P_in, P_out], axis=1)
    elif method == "DD":
        return P_in - P_out
    elif method == "SD":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return P_in - P_out
    else:
        #L2 distance between P_in and P_out
        return np.linalg.norm(P_in - P_out, axis=1)


def baseline(args):
    ''' The baseline from Yang Zhang's paper.
    I assume all the shadow models, and the shadow unlearning modes are available.
    I also assume the target model and the target unlearning model are available.
    I assume each shadow unlearning model (OUT model) and each target unlearning model are clearly associated with a sample in attack training/testing set

    For example:
    attack training set contains 5 samples, whose indexes are [1,2, 7, 16, 223]
    their labels are [1, 0, 1, 0, 1], where 1 means member, 0 means non-member
    there are one shadow model MS, and 5 shadow unlearning models MS_remove_1, MS_remove_2, MS_remove_7, MS_remove_16, MS_remove_223

    attack target set contains 5 samples, whose indexes are [3, 4, 5, 6, 8]
    their labels are [1, 0, 1, 0, 1], where 1 means member, 0 means non-member
    there are one target model MT, and 5 target unlearning models MT_remove_3, MT_remove_4, MT_remove_5, MT_remove_6, MT_remove_8


    1. Training:
        Given a set of attack training sample x, load IN-model and OUT-model, read posteriors Pin and Pout on x,
         construct attack feature by 5 methods. Train a classifier to distinguish members and non-members
    2. Testing:
        Given a set of attack targets x, construct attack feature and predict membership.
    '''

    # 1. Training
    '''

    X_train, ytest = "load attack training sample x" # todo: how to prepare attack training set?
    Pin_list = []
    Pout_list = []
    for i in range(len(X_train)):
        x = X_train[i] # x can be the index of the training sample in the dataset
        inModel = f"load shadow model"
        outModel = f"load shadow_model_remove_{x}"
        Pin = inModel(x)
        Pout = outModel(x)
        Pin_list.append(Pin.detach().numpy())
        Pout_list.append(Pout.detach().numpy())

    Pin = np.array(Pin_list)
    Pout = np.array(Pout_list)'''

    # prep data
    Pin, Pout, Pin_test, Pout_test, mem_train, mem_test = baseline_prep(args)

    # drop the second dimension
    Pin = Pin[:, 0, :]
    Pout = Pout[:, 0, :]
    Pin_test = Pin_test[:, 0, :]
    Pout_test = Pout_test[:, 0, :]
    if args['base_num_class'] == 2:
        train_index = np.where(np.array(mem_train) != 2)[0]
        test_index = np.where(np.array(mem_test) != 2)[0]

        Pin = Pin[train_index]
        Pout = Pout[train_index]

        Pin_test = Pin_test[test_index]
        Pout_test = Pout_test[test_index]

        mem_train = np.array(mem_train)[train_index]
        mem_test = np.array(mem_test)[test_index]

    for method in ["DC", "SC", "DD", "SD", "ED"]:
        attack_X = attack_feature_base(Pin, Pout, method="DC")
        # only 100 shadow model

        #attack_X = np.concatenate([attack_X[:100], attack_X[5000: 5100]], axis=0)

        #ytest = mem_train[:100] + mem_train[5000: 5100]
        ytest = mem_train

        ss = StandardScaler()
        attack_X = ss.fit_transform(attack_X)
        # split train and val (0.2)
        Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=42)
        # train a classifier

        clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=args['base_num_class'], layer_list=[200, 200], device=torch.device('cuda:0'))
        #clf.criterion = torch.nn.CrossEntropyLoss()
        # three class classification
        clf.criterion = torch.nn.CrossEntropyLoss()
        clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
        clf.to(torch.device('cuda:0'))
        clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=200, max_patient=20)

        # 2. Testing
        attack_X_test = attack_feature_base(Pin_test, Pout_test, method="DC")
        y_target = mem_test
        attack_X_test = ss.transform(attack_X_test)
        acc = clf.acc(attack_X_test, y_target)
        probs = clf.pred(attack_X_test)
        # acc by labels:
        y_pred = np.argmax(probs, axis=1)
        df_res = pd.DataFrame({"y_pred": y_pred, "y_target": y_target, "mem_label": mem_test})
        if args['base_num_class'] == 3:
            print("Method: ", method)
            print("ACC: %.4f" % acc)
            print("ACC by labels: ", df_res.groupby("mem_label").apply(lambda x: metrics.accuracy_score(x.y_target, x.y_pred)))
        else:
            performance = clf.all_metrics(attack_X_test, y_target)
            print("Method: ", method)
            print("ACC: %.4f, Prec: %.4f, Recall: %.4f, AUC: %.4f, F1: %.4f, TPR@0.01FPR: %.4f, TPR@0.05FPR: %.4f, TPR@0.10FPR: %.4f" % tuple(performance))


def baseline_prep(args):
    train_data, test_data = get_data(args['dataset_name'])

    target_m,  shadow_m, shadow_um = split_dataset(
        train_data, args['random'])

    in_list_test = []
    out_list_test = []
    mem_test = []
    in_list = []
    out_list = []
    mem_train = []
    # members
    for t in tqdm(range(min(args['trials'], 100)), desc='preparing members'): #todo: remove limit

        train_data, test_data = get_data(args['dataset_name'])
        save_path = os.getcwd() + f"/save/{args['net_name']}/scratch/{args['dataset_name']}2/{args['num_epochs']}/{args['lr']}/{t}//"
        if os.path.exists(save_path) is False:
            print("skip! No such file or directory: ", save_path)
            continue
        original_model_para = torch.load(f"{save_path}/original_model.pt")
        original_model = DNN(args)
        original_model.load_state_dict(original_model_para)
        unlearn_model_para = torch.load(f"{save_path}/unlearned_model.pt")
        unlearn_model = DNN(args)
        unlearn_model.load_state_dict(unlearn_model_para)
        target_sample = torch.load(f"{save_path}/target_sample.pth")[0][0]

        target_sample = torch.unsqueeze(target_sample.to(torch.float32), 0).to(args['device'])

        # load shadow models and shadow unlearning models
        #for sind in range(100):
        shadow_in = torch.load(f"{save_path}/original_model_shadow.pt")
        shadow_in_model = DNN(args)
        shadow_in_model.load_state_dict(shadow_in)
        shadow_in_model.model.to(shadow_in_model.device)
        shadow_out = torch.load(f"{save_path}/unlearned_model_shadow.pt")
        shadow_out_model = DNN(args)
        shadow_out_model.load_state_dict(shadow_out)
        shadow_out_model.model.to(shadow_out_model.device)
        Pin = shadow_in_model.predict_proba(target_sample)
        Pout = shadow_out_model.predict_proba(target_sample)
        in_list.append(Pin)
        out_list.append(Pout)
        mem_train.append(1)
        # non-member
        target_sample_nm, _ = sample_target_samples(test_data, args['proportion_of_group_unlearn'], args['dataset_name'])
        target_sample_nm = torch.unsqueeze(target_sample_nm.to(torch.float32), 0).to(args['device'])
        Pin_nm = shadow_in_model.predict_proba(target_sample_nm)
        Pout_nm = shadow_out_model.predict_proba(target_sample_nm)
        in_list.append(Pin_nm)
        out_list.append(Pout_nm)
        mem_train.append(0)
        # member but not unlearned
        # todo: is this member not unlearned correct?
        target_sample_mnu, _ = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'], args['dataset_name'])
        target_sample_mnu = torch.unsqueeze(target_sample_mnu.to(torch.float32), 0).to(args['device'])
        Pin_mnu = shadow_in_model.predict_proba(target_sample_mnu)
        Pout_mnu = shadow_out_model.predict_proba(target_sample_mnu)
        in_list.append(Pin_mnu)
        out_list.append(Pout_mnu)
        mem_train.append(2)




        Pin_test = original_model.predict_proba(target_sample)
        Pout_test = unlearn_model.predict_proba(target_sample)

        in_list_test.append(Pin_test)
        out_list_test.append(Pout_test)
        mem_test.append(1)
        # non-member
        Pin_test_nm = original_model.predict_proba(target_sample_nm)
        Pout_test_nm = unlearn_model.predict_proba(target_sample_nm)
        in_list_test.append(Pin_test_nm)
        out_list_test.append(Pout_test_nm)
        mem_test.append(0)
        # member but not unlearned
        target_sample_mnu, _ = sample_target_samples(train_data, args['proportion_of_group_unlearn'], args['dataset_name'])
        target_sample_mnu = torch.unsqueeze(target_sample_mnu.to(torch.float32), 0).to(args['device'])
        Pin_test_mnu = original_model.predict_proba(target_sample_mnu)
        Pout_test_mnu = unlearn_model.predict_proba(target_sample_mnu)
        in_list_test.append(Pin_test_mnu)
        out_list_test.append(Pout_test_mnu)
        mem_test.append(2)

    ## In-In set: the memeber that is not unlearned





    # non-members
    # there is no out model, out model is as same as out model
    # the targets are sampled from target_um
    # given each shadow model, the target_um is always non-member

    '''    for t in tqdm(range(min(args['trials'], 100)), desc='preparing non-members'):
        # sample target_um
        target_sample, _ = sample_target_samples(test_data,1,args['dataset_name'])[0][0]
        target_sample = torch.unsqueeze(target_sample.to(torch.float32), 0).to(args['device'])

        in_model_para = torch.load(f"{save_path}/original_model.pt")
        in_model = DNN(args)
        in_model.load_state_dict(in_model_para)
        out_model_para = torch.load(f"{save_path}/unlearned_model.pt")
        out_model = DNN(args)
        out_model.load_state_dict(out_model_para)

        #for sind in range(100):
        shadow_in = torch.load(f"{save_path}/original_model_shadow.pt")
        shadow_in_model = DNN(args)
        shadow_in_model.load_state_dict(shadow_in)
        shadow_in_model.model.to(shadow_in_model.device)
        shadow_out = torch.load(f"{save_path}/unlearned_model_shadow.pt")
        shadow_out_model = DNN(args)
        shadow_out_model.load_state_dict(shadow_out)
        shadow_out_model.model.to(shadow_out_model.device)
        Pin = shadow_in_model.predict_proba(target_sample)
        Pout = shadow_out_model.predict_proba(target_sample)
        in_list.append(Pin)
        out_list.append(Pout)
        mem_train.append(0)

        Pin_test = in_model.predict_proba(target_sample)
        Pout_test = out_model.predict_proba(target_sample)
        in_list_test.append(Pin_test)
        out_list_test.append(Pout_test)
        mem_test.append(0)'''
    Pin = np.array(in_list)
    Pout = np.array(out_list)
    Pin_test = np.array(in_list_test)
    Pout_test = np.array(out_list_test)
    return Pin, Pout, Pin_test, Pout_test, mem_train, mem_test



#
# if __name__ == '__main__':
#     args = parameter_parser()
#     baseline(args)

