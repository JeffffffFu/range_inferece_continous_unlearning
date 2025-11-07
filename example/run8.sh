
#python main.py --U_method GA --dataset_name sst5 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0
#python main.py --U_method GA --dataset_name news20 --net_name pythia70m --batch_size 32 --lr 0.00001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:3

python main.py --U_method GA --dataset_name mnli --net_name pythia70m --batch_size 128 --lr 0.0001  --trials 3 --proportion_of_group_unlearn 64 --num_epochs 10  --device cuda:0
