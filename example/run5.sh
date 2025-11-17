
# retrain densenet
#python main.py --pre_train both --U_method retrain --dataset_name cifar10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2
#python main.py --pre_train both --U_method retrain --dataset_name cifar100 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2
#python main.py --pre_train both --U_method retrain --dataset_name tinyimagenet --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2


# sisa densenet
#python main.py --pre_train both --U_method sisa --dataset_name cifar10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2
#python main.py --pre_train both --U_method sisa --dataset_name cifar100 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2
#python main.py --pre_train both --U_method sisa --dataset_name tinyimagenet --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method sisa --dataset_name cinic10 --net_name densenet --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3  --observations 5  --device cuda:2


#scrub
#python main.py --pre_train both --U_method scrub --dataset_name cifar10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cifar100 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name tinyimagenet --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train both --U_method scrub --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2


#well-over  retrain_save_target_for_population_attack_batch2
#python main.py --pre_train target --U_method retrain --dataset_name cifar10 --net_name simple_cnn_dropout --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:3
#python main.py --pre_train target --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 3 --observations 5  --device cuda:2
#python main.py --pre_train target --U_method retrain --dataset_name cifar10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1 --observations 5  --device cuda:2
#python main.py --pre_train target --U_method retrain --dataset_name cinic10 --net_name simple_cnn --num_epochs 50 --proportion_of_group_unlearn 0.02 --trials 1 --observations 5  --device cuda:2

#python main.py --pre_train both --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50  --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.25 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.5 --trials 3 --observations 5  --device cuda:0
#python main.py --pre_train shadow --U_method retrain --dataset_name cinic10 --net_name resnet18 --num_epochs 50 --size_of_shadow_training 0.75 --trials 3 --observations 5  --device cuda:0
#

#python main.py --U_method NPO --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0
#python main.py --U_method GA --dataset_name mrpc --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 16 --num_epochs 10  --device cuda:0
python main.py --U_method GA --dataset_name sst5 --net_name opt13b --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 5  --device cuda:0
