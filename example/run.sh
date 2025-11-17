# for population_improve

#retrain resnet18
python main.py  --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

##retrain densenet
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#
##retrain simple_cnn
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

#
##sisa  resnet18
python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sisa --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sisa --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

##
###sisa densenet
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#
##sisa simple_cnn
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sisa --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

#
###GA resnet18
python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method GA --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method GA --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

###GA simple_cnn
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#
####GA densenet
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#

###scrub resnet18
python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method scrub --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method scrub --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method scrub --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0


##sparsity resnet18
python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
python main.py --attack_method TC_MIA --U_method sparsity --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cinic10 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name tinyimagenet --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cinic10 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name tinyimagenet --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0


##NegGrad resnet18
#python main.py --attack_method TC_MIA --U_method NegGrad --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method NegGrad --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method NegGrad --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method NegGrad --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#

# different confidence and ASR samples cifar10,
#python main.py --attack_method TC_MIA --U_method retrain --flag random --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag low_conf --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag high_conf --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag low_entropy --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag high_entropy --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag in --dataset_name cifar10 --net_name resnet18  --trials 5 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag out --dataset_name cifar10 --net_name resnet18  --trials 5 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

# different confidence and ASR samples cinic10
#python main.py --attack_method TC_MIA --U_method retrain --flag random --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag low_conf --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag high_conf --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag low_entropy --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag high_entropy --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag low_asr --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag high_asr --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag in --dataset_name cinic10 --net_name resnet18  --trials 5 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag out --dataset_name cinic10 --net_name resnet18  --trials 5 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0


# transfer model, need to fix the network in shadow model
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name simple_cnn  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name densenet  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

# transfer dataset, need to fix the dataset in shadow model
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

# transfer algorithm, need to fix the algorithm in shadow model
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method GA --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method sparsity --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0


# size of forget set  need to fix the shadow proportion_of_group_unlearn cifar100
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.01 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.05 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.1 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.3 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.4 --observations 5 --device cuda:0

#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.1 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.2 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.3 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.4 --observations 5 --device cuda:0


# size of training set of attack model, change the observations

#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 1 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 3 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 8 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 10 --device cuda:0


#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 1 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 3 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 8 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 10 --device cuda:0


# size of shadow training set
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --size_of_shadow_training 0.1 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --size_of_shadow_training 0.25 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --size_of_shadow_training 0.5 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --size_of_shadow_training 0.75 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --size_of_shadow_training 0.1 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --size_of_shadow_training 0.25 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --size_of_shadow_training 0.5 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name tinyimagenet --net_name resnet18  --trials 3 --size_of_shadow_training 0.75 --observations 5 --device cuda:0

# different Size Ratios, need to add adjust_ratio_samples() in in TS-MIA
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 5 --device cuda:0



# over-to-well cifar10 ,find optimal features
#python main.py --attack_method TC_MIA --U_method retrain --flag over-well --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag well-over --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag well-well --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag over-over --dataset_name cifar10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

# # over-to-well cinic10
#python main.py --attack_method TC_MIA --U_method retrain --flag over-well --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag well-over --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag well-well --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --flag over-over --dataset_name cinic10 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

# DP-retrain cifar10
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_0.5 --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_1.0 --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_1.5 --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_2.0 --dataset_name cifar10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
# DP-retrain cinic10
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_1.0 --dataset_name cinic10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain_dp --flag dp_2.0 --dataset_name cinic10 --net_name simple_cnn  --trials 1 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0

