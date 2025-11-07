
## U_Leak
#retrain resnet18
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

##retrain densenet
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#
##retrain simple_cnn
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method retrain --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

#
##sisa resnet18
python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar10 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar100 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sisa --dataset_name cinic10 --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sisa --dataset_name tinyimagenet --net_name resnet18  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

#
##sisa densenet
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar10 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar100 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cinic10 --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name tinyimagenet --net_name densenet  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#
##sisa simple_cnn
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cifar100 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name cinic10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sisa --dataset_name tinyimagenet --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

###GA resnet18
python main.py --attack_method U_Leak --U_method GA --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method GA --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method GA --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method GA --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

##GA simple_cnn
#python main.py --attack_method U_Leak --U_method GA --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

###GA densenet
#python main.py --attack_method U_Leak --U_method GA --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method GA --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2


###scrub resnet18
python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method scrub --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method scrub --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

###scrub simple_cnn
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#
####scrub densenet
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method scrub --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2


##sparsity resnet18
python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sparsity --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
python main.py --attack_method U_Leak --U_method sparsity --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method sparsity --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

##NegGrad resnet18
#python main.py --attack_method U_Leak --U_method NegGrad --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method NegGrad --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method NegGrad --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2
#python main.py --attack_method U_Leak --U_method NegGrad --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:2

###scrub simple_cnn train
#python main.py --pre_train both --U_method scrub --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cifar100 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name tinyimagenet --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#
####scrub simple_cnn train
#python main.py --pre_train both --U_method scrub --dataset_name cifar10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cifar100 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name tinyimagenet --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cinic10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#
