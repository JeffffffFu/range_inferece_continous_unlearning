# save
# retrain resnet18 proportion_of_group_unlearn 0.02
#python main.py --pre_train both --U_method retrain --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:0


#python main.py --attack_method TC_MIA --pre_train both --U_method retrain --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method U_Leak --pre_train both --U_method retrain --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method Double_Attack --pre_train both --U_method retrain --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0

#python main.py --attack_method TC_MIA --pre_train both --U_method GA --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method U_Leak --pre_train both --U_method GA --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method Double_Attack --pre_train both --U_method GA --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0



#python main.py --attack_method TC_MIA --pre_train both --U_method NPO --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method U_Leak --pre_train both --U_method NPO --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --attack_method Double_Attack --pre_train both --U_method NPO --dataset_name news20 --net_name pythia70m --num_epochs 20 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0

# sisa resnet18 proportion_of_group_unlearn 0.02
#python main.py --pre_train both --U_method sisa --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sisa --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sisa --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sisa --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sisa --dataset_name tinyimagenet --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#

#sparsity
#python main.py --pre_train both --U_method sparsity --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sparsity --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sparsity --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method sparsity --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0




###sparsity resnet18 train
#python main.py --pre_train both --U_method sparsity --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#

#sparsity simple_cnn train
#python main.py --pre_train both --U_method sparsity --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cifar100 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name tinyimagenet --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3

###sparsity densenet train
#python main.py --pre_train both --U_method sparsity --dataset_name cifar10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cifar100 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name tinyimagenet --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train both --U_method sparsity --dataset_name cinic10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3


# For SST5
#python main.py --pre_train both --U_method retrain --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method GA --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method NPO --dataset_name sst5 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0

# For news20
#python main.py --pre_train both --U_method retrain --dataset_name news20 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train shadow --U_method GA --dataset_name news20 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train both --U_method NPO --dataset_name news20 --net_name pythia70m --num_epochs 10 --batch_size 32 --lr 0.00001 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:0


python main.py --U_method npo --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0
