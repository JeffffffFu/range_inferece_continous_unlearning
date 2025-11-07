#python main.py --U_method continuous_unlearn_retrain --dataset_name cifar100 --net_name resnet18  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 50  --device cuda:0
#python main.py --U_method continuous_unlearn_NPO --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:3
python main.py --U_method GA --dataset_name snli --net_name pythia70m --batch_size 128 --lr 0.0001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0
