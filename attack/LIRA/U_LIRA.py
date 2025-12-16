import torch
import torch.nn.functional as F
import os

from data.load_data import get_data
from data.prepare_data import split_dataset
from model.DNN import DNN
import numpy as np

def U_LIRA(
        x, #target sample
        original_model, #before unlearning
        unlearned_model, #: after unlearned
        original_model_shadow,
        unlearned_model_shadow,
        shadow_um,
        args, #: arguments
):

    # 初始化存储变量
    O_in = []
    O_out = []

    for t in range(args['observations']):
       # print(f'The {t+1}-th observations----------')
        x_prime, _ = sample_target_samples(shadow_um, args['proportion_of_group_unlearn'], args['dataset_name'])

        o = original_model_shadow.logits(x) - unlearned_model_shadow.logits(x)
        #o = original_model.logits(x) - unlearned_model.logits(x)

        O_in.append(o)

        o_prime = original_model_shadow.logits(x_prime) - unlearned_model_shadow.logits(x_prime)
       # o_prime = original_model.logits(x_prime) - unlearned_model.logits(x_prime)

        O_out.append(o_prime)

    # 拟合 O 的高斯分布
    O_in_tensor = torch.tensor(O_in)
    mu_in, sigma_in = O_in_tensor.mean(), O_in_tensor.std()

    # 拟合 Ô 的高斯分布
    O_out_tensor = torch.tensor(O_out)
    mu_out, sigma_out = O_out_tensor.mean(), O_out_tensor.std()
    # 最后一步计算成员关系概率
    a = original_model.logits(x) - unlearned_model.logits(x)
    sigma_out = torch.tensor(1.0)
    print(mu_in,sigma_out)
    print(mu_in,sigma_out)
    print(a)

    p_member = gaussian_pdf(a, mu_in, sigma_in) / (
            gaussian_pdf(a, mu_in, sigma_in) + gaussian_pdf(a, mu_out, sigma_out)
    )

    return p_member

def U_LIRA_load(
        t,
        args, #: arguments
        infer_member
):

    # 初始化存储变量
    O_in = []
    O_out = []
    load_path = os.getcwd() + f"/save/{args['net_name']}/{args['U_method']}/{args['dataset_name']}/{args['num_epochs']}/{args['lr']}/{t}//"
    device= torch.load(f'{load_path}/device.pth')
    args['device'] = device
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    random = torch.load(f'{load_path}/random_seed.pth')
    target_m,  shadow_m, shadow_um = split_dataset(train_data, random)

    if infer_member:
        target_sample=torch.load(f'{load_path}/forgot_sample.pth')

    else:  #unseen
        target_sample=torch.load(f'{load_path}/unseen_sample.pth')


    original_model = DNN(args).to(device)
    original_model.load_state_dict(torch.load(f'{load_path}/target_original_model.pt'))
    unlearned_model = DNN(args).to(device)
    unlearned_model.load_state_dict(torch.load(f'{load_path}/target_unlearned_model.pt'))


    for i in range(args['observations']):
       # print(f'The {t+1}-th observations----------')
       if infer_member: #infer forget samples
           load_path_shadow = f"{load_path}/forgot_sample/shadow_{i}//"
           original_model_shadow_forget = DNN(args).to(device)
           original_model_shadow_forget.load_state_dict(torch.load(f'{load_path_shadow}/original_model_shadow_forgot.pt'))
           unlearned_model_shadow_forget = DNN(args).to(device)
           unlearned_model_shadow_forget.load_state_dict(torch.load(f'{load_path_shadow}/unlearned_model_shadow_forgot.pt'))
           original_model_shadow_unseen = DNN(args).to(device)
           original_model_shadow_unseen.load_state_dict(torch.load(f'{load_path_shadow}/original_model_shadow_unseen.pt'))
           unlearned_model_shadow_unseen = DNN(args).to(device)
           unlearned_model_shadow_unseen.load_state_dict(torch.load(f'{load_path_shadow}/unlearned_model_shadow_unseen.pt'))

       else:  #infer unseen samples
           load_path_shadow = f"{load_path}/unseen_sample/shadow_{i}//"
           original_model_shadow_forget = DNN(args).to(device)
           original_model_shadow_forget.load_state_dict(torch.load(f'{load_path_shadow}/original_model_shadow_forgot.pt'))
           unlearned_model_shadow_forget = DNN(args).to(device)
           unlearned_model_shadow_forget.load_state_dict(torch.load(f'{load_path_shadow}/unlearned_model_shadow_forgot.pt'))
           original_model_shadow_unseen = DNN(args).to(device)
           original_model_shadow_unseen.load_state_dict(torch.load(f'{load_path_shadow}/original_model_shadow_unseen.pt'))
           unlearned_model_shadow_unseen = DNN(args).to(device)
           unlearned_model_shadow_unseen.load_state_dict(torch.load(f'{load_path_shadow}/unlearned_model_shadow_unseen.pt'))


       original_model_shadow_forget.eval()
       unlearned_model_shadow_forget.eval()
       o = original_model_shadow_forget.logits(target_sample) - unlearned_model_shadow_forget.logits(target_sample)
      # O_in.append(o)
       O_in.append(abs(o))

       original_model_shadow_unseen.eval()
       unlearned_model_shadow_unseen.eval()
       o_prime = original_model_shadow_unseen.logits(target_sample) - unlearned_model_shadow_unseen.logits(target_sample)
      # O_out.append(o_prime)
       O_out.append(abs(o_prime))

   # 拟合 O 的高斯分布
    O_in = np.array(O_in)
    mu_in, sigma_in = np.mean(O_in), np.std(O_in)

   # 拟合 Ô 的高斯分布
    O_out = np.array(O_out)
    mu_out, sigma_out = np.mean(O_out), np.std(O_out)

   # 最后一步计算成员关系概率
   # a = original_model.logits(target_sample) - unlearned_model.logits(target_sample)
    a = abs(original_model.logits(target_sample) - unlearned_model.logits(target_sample))
    print("shadow_forgot_model:",mu_in, sigma_in)
    print("shadow_unseen_model:",mu_out, sigma_out)
    print("target:",a)

    p_member = gaussian_pdf(a, mu_in, sigma_in) / (
           gaussian_pdf(a, mu_in, sigma_in) + gaussian_pdf(a, mu_out, sigma_out)
    )

    return p_member




def U_LIRA_load_for_three_domain(
        t,
        args, #: arguments
        infer
):

    # distinguish three worlds
    # 初始化存储变量
    O_in = []
    O_out = []
    O_retain=[]
    load_path = os.getcwd() + f"/save/{args['net_name']}/{args['U_method']}/{args['dataset_name']}/{args['num_epochs']}/{args['lr']}/{t}//"
    device = torch.load(f'{load_path}/device.pth')
    args['device'] = device
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    random = torch.load(f'{load_path}/random_seed.pth')
    target_m,  shadow_m, shadow_um = split_dataset(train_data, random)

    if infer==2:
        target_sample = torch.load(f'{load_path}/forgot_sample.pth')

    elif infer==0:  # unseen
        # target_sample, _ = sample_target_samples(target_um,1,args['dataset_name'])
        #  target_sample, _ = sample_target_samples(target_m,1,args['dataset_name'])
        target_sample = torch.load(f'{load_path}/unseen_sample.pth')
    else:  # retain
        target_sample = torch.load(f'{load_path}/retain_sample.pth')

    original_model = DNN(args).to(device)
    original_model.load_state_dict(torch.load(f'{load_path}/target_original_model.pt'))
    unlearned_model = DNN(args).to(device)
    unlearned_model.load_state_dict(torch.load(f'{load_path}/target_unlearned_model.pt'))

    for i in range(args['observations']):
        # print(f'The {t+1}-th observations----------')
        if infer==2:  # infer forget samples
            load_path_shadow = f"{load_path}/forgot_sample/shadow_{i}//"
            original_model_shadow_forget = DNN(args).to(device)
            original_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_forgot.pt'))
            unlearned_model_shadow_forget = DNN(args).to(device)
            unlearned_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_forgot.pt'))
            original_model_shadow_retain = DNN(args).to(device)
            original_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_retain.pt'))
            unlearned_model_shadow_retain = DNN(args).to(device)
            unlearned_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_retain.pt'))
            original_model_shadow_unseen = DNN(args).to(device)
            original_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_unseen.pt'))
            unlearned_model_shadow_unseen = DNN(args).to(device)
            unlearned_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_unseen.pt'))

        elif infer==0:  # infer unseen samples
            load_path_shadow = f"{load_path}/unseen_sample/shadow_{i}//"
            original_model_shadow_forget = DNN(args).to(device)
            original_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_forgot.pt'))
            unlearned_model_shadow_forget = DNN(args).to(device)
            unlearned_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_forgot.pt'))
            original_model_shadow_retain = DNN(args).to(device)
            original_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_retain.pt'))
            unlearned_model_shadow_retain = DNN(args).to(device)
            unlearned_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_retain.pt'))
            original_model_shadow_unseen = DNN(args).to(device)
            original_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_unseen.pt'))
            unlearned_model_shadow_unseen = DNN(args).to(device)
            unlearned_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_unseen.pt'))

        else:
            load_path_shadow = f"{load_path}/retain_sample/shadow_{i}//"
            original_model_shadow_forget = DNN(args).to(device)
            original_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_forgot.pt'))
            unlearned_model_shadow_forget = DNN(args).to(device)
            unlearned_model_shadow_forget.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_forgot.pt'))
            original_model_shadow_retain = DNN(args).to(device)
            original_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_retain.pt'))
            unlearned_model_shadow_retain = DNN(args).to(device)
            unlearned_model_shadow_retain.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_retain.pt'))
            original_model_shadow_unseen = DNN(args).to(device)
            original_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/original_model_shadow_unseen.pt'))
            unlearned_model_shadow_unseen = DNN(args).to(device)
            unlearned_model_shadow_unseen.load_state_dict(
                torch.load(f'{load_path_shadow}/unlearned_model_shadow_unseen.pt'))

        original_model_shadow_forget.eval()
        unlearned_model_shadow_forget.eval()
        o = original_model_shadow_forget.logits(target_sample) - unlearned_model_shadow_forget.logits(target_sample)
       # O_in.append(o)
        O_in.append(abs(o))

        original_model_shadow_unseen.eval()
        unlearned_model_shadow_unseen.eval()
        o_prime = original_model_shadow_unseen.logits(target_sample) - unlearned_model_shadow_unseen.logits(target_sample)
      #  O_out.append(o_prime)
        O_out.append(abs(o_prime))


        original_model_shadow_retain.eval()
        unlearned_model_shadow_retain.eval()
        o_retain = original_model_shadow_retain.logits(target_sample) - unlearned_model_shadow_retain.logits(target_sample)
       # O_retain.append(o_retain)
        O_retain.append(abs(o_retain))


    O_in = np.array(O_in)
    mu_in, sigma_in = np.mean(O_in), np.std(O_in)

   # 拟合 Ô 的高斯分布
    O_out = np.array(O_out)
    mu_out, sigma_out = np.mean(O_out), np.std(O_out)

    # 拟合 Ô 的高斯分布
    O_retain = np.array(O_retain)
    mu_retain, sigma_retain = np.mean(O_retain), np.std(O_retain)

    # 最后一步计算成员关系概率
#    a = original_model.logits(target_sample) - unlearned_model.logits(target_sample)
    a = abs(original_model.logits(target_sample) - unlearned_model.logits(target_sample))

    print(mu_in, sigma_in)
    print(mu_out, sigma_out)
    print(mu_retain, sigma_retain)
    print(a)

    p_forgot = gaussian_pdf(a, mu_in, sigma_in)
    p_unseen =gaussian_pdf(a, mu_out, sigma_out)
    p_retain =gaussian_pdf(a, mu_retain, sigma_retain)

    p_list=[p_unseen, p_retain, p_forgot]
    index=p_list.index(max(p_list))

    return index


def gaussian_pdf(x, mean, std):
    """
    高斯概率密度函数
    """
    # 防止除以零的情况
    if std == 0:
        raise ValueError("Standard deviation (std) cannot be zero.")

    two_pi = 2 * np.pi

    return (1.0 / (std * np.sqrt(two_pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


