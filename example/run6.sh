# retrain simple_cnn
#python main.py --pre_train both --U_method retrain --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name tinyimagenet --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:3

# sisa simple_cnn
#python main.py --pre_train both --U_method sisa --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cifar100 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name tinyimagenet --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cinic10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:3

#GA resnet18
#python main.py --pre_train both --U_method GA --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method GA --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method GA --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method GA --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3


# size of forget set cifar100,cinic10,Percentage of Unlearned Samples, retrain_save_target_for_population_attack
# when attack , fix shadow to retrain,0.02
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.1 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.2 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.3 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.4 --trials 3 --observations 5  --device cuda:3

#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.1 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.2 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.3 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.4 --trials 3 --observations 5  --device cuda:3


#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.2 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.3 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.4 --trials 3 --observations 5  --device cuda:0

# for example, attack as follows
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.2 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.3 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.4 --trials 3 --observations 5  --device cuda:0

#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.1 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.2 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.3 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.4 --trials 3 --observations 5  --device cuda:0



# Size of attack training set cifar100,cinic10,sst5, just need more shadow training  retrain_save_shadow_for_population_attack_batch
# when attack , just adjust the observation times. 5 is 30% of shadow training set, 10 is 60% of shadow training set. 1,3,5,8,10
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:0.

 # for example, attack as follows:
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 1  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 3  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 8  --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 10  --device cuda:0




#size of shadow set cifar100,cinic10, retrain_save_shadow_for_population_attack2
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.1 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.25 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.5 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.75 --trials 3 --observations 5  --device cuda:3

#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.1 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.25 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.5 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.75 --trials 3 --observations 5  --device cuda:3


#different Size Ratios of training set, cifar100,cinic10, need to add adjust_ratio_samples() in in TS-MIA
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cifar100 --net_name resnet18  --trials 3  --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3  --observations 5 --device cuda:0
#python main.py --attack_method TC_MIA --U_method retrain --dataset_name sst5 --net_name pythia70m  --trials 3  --observations 5 --device cuda:0


#differential confidence samples   retrain_save_target_for_population_attack_batch3
#python main.py --pre_train shadow --U_method retrain --dataset_name cifar10 --net_name resnet18 --num_epochs 55 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 45 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3

# differential ASR samples  retrain_save_target_for_population_attack_batch4
#python main.py --pre_train target --U_method retrain --dataset_name cifar10 --net_name resnet18 --num_epochs 55 --proportion_of_group_unlearn 0.02 --trials 5 --observations 5  --device cuda:2
#python main.py --pre_train target --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 45 --proportion_of_group_unlearn 0.02 --trials 5 --observations 5  --device cuda:2


# outliner samples  retrain_save_target_for_population_attack_batch5
#python main.py --pre_train target --U_method retrain --dataset_name cifar10 --net_name resnet18 --num_epochs 55 --proportion_of_group_unlearn 0.02 --trials 5 --observations 5  --device cuda:2
#python main.py --pre_train target --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 45 --proportion_of_group_unlearn 0.02 --trials 5 --observations 5  --device cuda:2


# pearson corrleation
#python main.py --attack_method None --pre_train target --U_method retrain --dataset_name cifar10 --net_name resnet18  --trials 3 --num_epochs 50 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0
#python main.py --attack_method None --pre_train target --U_method retrain --dataset_name cinic10 --net_name resnet18  --trials 3 --num_epochs 30 --proportion_of_group_unlearn 0.02 --observations 5 --device cuda:0


#python main.py --pre_train both --U_method sisa --dataset_name cifar100 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1  --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cifar100 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1  --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name tinyimagenet --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sisa --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1  --observations 5  --device cuda:3
#python main.py --pre_train target --U_method retrain --dataset_name cinic10 --net_name simple_cnn_dropout --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0

python main.py --U_method GA --dataset_name sst5 --net_name roberta --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0

