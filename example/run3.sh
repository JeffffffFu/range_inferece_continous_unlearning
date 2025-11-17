##Double Attack
#retrain resnet18
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3


##retrain densenet
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
##retrain simple_cnn
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method retrain --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
#
###sisa resnet18
python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar10 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar100 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sisa --dataset_name cinic10 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sisa --dataset_name tinyimagenet --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

##
###sisa densenet
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar10 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar100 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cinic10 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name tinyimagenet --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
#sisa simple_cnn
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cifar100 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name cinic10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sisa --dataset_name tinyimagenet --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

###GA resnet18
python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method GA --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method GA --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

###GA simple_cnn
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
####GA densenet
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method GA --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3


###scrub resnet18
python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method scrub --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method scrub --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method scrub --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3


##sparsity resnet18
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
python main.py --attack_method Double_Attack --U_method sparsity --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method sparsity --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3

#NegGrad resnet18
#python main.py --attack_method Double_Attack --U_method NegGrad --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method NegGrad --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method NegGrad --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3
#python main.py --attack_method Double_Attack --U_method NegGrad --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:3


