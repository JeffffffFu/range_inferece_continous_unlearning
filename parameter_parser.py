import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        choices=[ 'mnist', 'fmnist', 'mnist2','cifar10', 'stl10', 'cifar100','svhn','celebA','tinyimagenet','cinic10','sst5','news20','snli','mnli','mrpc','imdb','rte','ag_news'])
    parser.add_argument('--original_label', type=str, default='NY',
                        choices=['income', 'severity', 'LA', 'NY', 'default'])
    # parser.add_argument('--exp', type=str, default='mem_inf',
    #                     choices=['model_train', 'mem_inf'],
    #                     help="'mem_train' train the original-unlearning model pairs, 'mem_inf' launch the attack")
    parser.add_argument('--device', type=str, default='cuda:3',
                        help="Choose the  device")
    parser.add_argument('--flag', type=str, default='none',
                        help="differential confidence samples")
    parser.add_argument('--random', type=int, default=0)
    ######################### target model related parameters ################################
    parser.add_argument('--net_name', type=str, default='resnet18',
                        choices=['DT', 'MLP', 'LR', 'RF', 'LRTorch', 'MLPTorch', 'simple_cnn', 'resnet18', 'resnet20','vgg','resnet18_dp','resnet50', 'densenet','CNN_MNIST','simple_cnn_dropout','pythia70m','pythia70m_dropout','roberta','opt13b'])
    # parser.add_argument('--attack_model', type=str, default='DT',
    #                     choices=['DT', 'MLP', 'LR', 'RF'])
    parser.add_argument('--U_method', type=str, default='None',
                        choices=['retrain', 'sisa','GA','sparsity','IF','fisher','scrub','sisa','retrain_dp','certified','NegGrad','NPO','None','continuous_unlearn_retrain','continuous_unlearn_NPO','continuous_unlearn_GA','all','continuous_update_finetune'])
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--pre_train', type=str, default='both',choices=['both','target', 'shadow'])
    parser.add_argument('--attack_method', type=str, default='None',
                        choices=['U_LIRA', 'TC_MIA','U_Leak','Double_Attack','None'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optim', type=str, default="Adam",
                        choices=['Adam', 'SGD'])
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: .1)", )
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="dropout rate for pythia70m_dropout model (default: 0.1)", )
    ######################### attack related parameters ################################
    parser.add_argument('--trials', type=int, default=3,
                        help="number of trials")
    parser.add_argument('--observations', type=int, default=5,
                        help="number of observations")
    parser.add_argument('--base_num_class', type=int, default=3,
                        help="number of class for baseline: 2 or 3")
    parser.add_argument('--proportion_of_group_unlearn', type=float, default=0.02,
                        help=">=1 mean the exact number of unlearn")
    parser.add_argument('--number_of_shadow_unlearned_model', type=int, default=1,
                        help="1 means number of shadow unlearned model equal to number of shadow original model")
    parser.add_argument('--size_of_shadow_training', type=float, default=-1,
                        help="-1 means using all shadow dataset for training")
    # For DPSGD
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="noise of DPSGD")
    parser.add_argument('--eps', type=float, default=1.0,
                        help="privacy budget of DPSGD")
    parser.add_argument('--C', type=float, default=1.0,
                        help="C of model parameters (for DPSGD)")
    #For certified removal

    parser.add_argument('--max_norm', type=float, default=-1,
                        help="max_norm of model parameters (for certified removal), -1 means doesn't clipping")

    # parser.add_argument('--shadow_set_num', type=int, default=10,
    #                     help="Number of shadow original model")
    # parser.add_argument('--shadow_set_size', type=int, default=2000,
    #                     help="Number of shadow model training samples")
    # parser.add_argument('--shadow_unlearning_size', type=int, default=20,
    #                     help="Number of unlearned model")
    # parser.add_argument('--shadow_unlearning_num', type=int, default=1,
    #                     help="Number of deleted records to generate unlearned model")
    # parser.add_argument('--shadow_num_shard', type=int, default=10,
    #                     help="Number of shards")
    #
    # parser.add_argument('--target_set_num', type=int, default=10,
    #                     help="Number of target original model")
    # parser.add_argument('--target_set_size', type=int, default=2000,
    #                     help="Number of target model training samples")
    # parser.add_argument('--target_unlearning_size', type=int, default=20,
    #                     help="Number of unlearned model")
    # parser.add_argument('--target_unlearning_num', type=int, default=1,
    #                     help="Number of deleted records to generate unlearned model")
    # parser.add_argument('--target_num_shard', type=int, default=10,
    #                     help="Number of shards")

    # parser.add_argument('--samples_to_evaluate', type=str, default="in_out",
    #                     choices=['in_in', 'in_out', 'in_out_multi_version'],
    #                     help="Samples to evaluate")
    ######################### defense related parameters ################################
    parser.add_argument('--top_k', type=int, default=4, choices=[0, 1, 2, 3, 4],
                        help=" 0 (label), 4 (no defense)")

    parser.add_argument("-c", "--max-per-sample-grad_norm", type=float, default=1.0, metavar="C",
                        help="Clip per-sample gradients to this norm (default 1.0)", )
    parser.add_argument("--delta", type=float, default=1e-5, metavar="D",
                        help="Target delta (default: 1e-5)", )
    parser.add_argument("-sr", "--sample-rate", type=float, default=0.001, metavar="SR",
                        help="sample rate used for batch construction (default: 0.001)", )
    parser.add_argument("--secure_rng", action="store_true", default=False,
                        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost", )

    parser.add_argument('--n_accumulation_steps', type=float, default=1)
    parser.add_argument('--is_dp_defense', type=bool, default=False)

    args = vars(parser.parse_args())
    #args = parser.parse_args()

    return args