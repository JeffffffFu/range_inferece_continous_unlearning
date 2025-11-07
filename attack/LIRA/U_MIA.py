import random

import numpy as np

from attack.LIRA.U_LIRA import U_LIRA, U_LIRA_load, U_LIRA_load_for_three_domain
from attack.metric.metric import  acc_for_three_domain
from data.load_data import get_data
from model.DNN import DNN
import  torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os



def U_MIA(args):
    if args['attack_method']=='U_LIRA':
        p_member_list = []
        for i in range(args['trials']):
            print(f"The {i}-th U_MIA----")
            original_model, unlearned_model, original_model_shadow, unlearned_model_shadow, target_sample, shadow_um = retrain_and_shadow_model( args)
            p_member=U_LIRA(target_sample,original_model,unlearned_model,original_model_shadow,unlearned_model_shadow,shadow_um,args)
            p_member_list.append(p_member)
        print(p_member_list)
    elif args['attack_method']=='baseline':
        pass
    else:
        print('Invalid attack method')
        return

def U_MIA_load(args):
    if args['attack_method']=='U_LIRA':
        p_member_scores = []
        p_nonmember_scores=[]
        y_member=[]
        y_nonmember=[]
       # for t in range(args['trials']):

        for t in tqdm(range(min(args['trials'], 50)), desc='U_MIA to member'):
            p_member=U_LIRA_load(t,args,True)
            p_member_scores.append(p_member)
            y_member.append(1)
            print(p_member_scores)
            count = sum(1 for x in p_member_scores if x > 0.5)
            print(count/len(y_member))
        for t in  tqdm(range(min(args['trials'], 50)), desc='U_MIA to nonmember'):
            p_nonmember=U_LIRA_load(t,args,False)
            p_nonmember_scores.append(p_nonmember)
            y_nonmember.append(0)
            print(p_nonmember_scores)
            count = sum(1 for x in p_nonmember_scores if x < 0.5)
            print(count/len(y_nonmember))



    elif args['attack_method']=='baseline':
        pass
    else:
        print('Invalid attack method')
        return

def U_MIA_load_three_domains(args):

    p_unseen=[]
    p_retain=[]
    p_forgot=[]

    y_unseen=[]
    y_retain=[]
    y_forgot=[]

    for t in tqdm(range(min(args['trials'], 50)), desc='U_MIA to unseen sample'):
        p = U_LIRA_load_for_three_domain(t, args, 'unseen')
        p_unseen.append(p)
        y_unseen.append(0)

    for t in tqdm(range(min(args['trials'], 50)), desc='U_MIA to retain sample'):
        p = U_LIRA_load_for_three_domain(t, args, 'retain')
        p_retain.append(p)
        y_retain.append(1)

    for t in tqdm(range(min(args['trials'], 50)), desc='U_MIA to forgot sample'):
        p = U_LIRA_load_for_three_domain(t, args, 'forgot')
        p_forgot.append(p)
        y_forgot.append(2)

    acc = acc_for_three_domain(y_unseen, y_retain,y_forgot, p_unseen, p_retain,p_forgot)

