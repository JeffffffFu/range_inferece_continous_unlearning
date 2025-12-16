import torch
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from data.cinic10 import fetch_cinic10, get_cinic10
from data.tinyimagenet import fetch_tinyimagenet, TinyImageNet

class SST5Dataset(Dataset):
    """SST5数据集类，用于文本情感分类"""
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label'] - 1
        assert 0 <= label <= 4, f'标签越界: {label}, 原始: {item["label"]}, 文本: {text}'
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class News20Dataset(Dataset):
    """20 Newsgroups数据集类，用于文本分类"""
    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        assert 0 <= label <= 19, f'标签越界: {label}, 文本: {text[:100]}...'
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextClassificationDataset(Dataset):
    """通用文本分类数据集类，用于单文本输入任务"""
    def __init__(self, dataset, tokenizer, text_field='text', label_field='label', max_length=256, label_transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.label_field = label_field
        self.max_length = max_length
        self.label_transform = label_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[self.text_field]
        label = item[self.label_field]
        if self.label_transform is not None:
            label = self.label_transform(label)
        
        # 确保标签是整数且在合理范围内
        label = int(label)
        if label < 0:
            raise ValueError(f"标签值不能为负数: {label}, 索引: {idx}")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class PairedTextClassificationDataset(Dataset):
    """通用文本分类数据集类，用于句对分类任务"""
    def __init__(self, dataset, tokenizer, text_field1='sentence1', text_field2='sentence2', label_field='label', max_length=256, label_transform=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_field1 = text_field1
        self.text_field2 = text_field2
        self.label_field = label_field
        self.max_length = max_length
        self.label_transform = label_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text1 = item[self.text_field1]
        text2 = item[self.text_field2]
        label = item[self.label_field]
        if self.label_transform is not None:
            label = self.label_transform(label)

        encoding = self.tokenizer(
            text1,
            text_pair=text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

SHAPES = {
    "cifar10": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1),
    "cifar100": (32, 32, 3),
    "svhn": (32, 32, 3),
    "celebA": (128, 128, 3),
    "cinic10": (32, 32, 3),
    "sst5": (128, 1, 1),
    "news20": (256, 1, 1),
    "snli": (256, 1, 1),
    "mnli": (256, 1, 1),
    "mrpc": (256, 1, 1),
    "imdb": (512, 1, 1),
    "rte": (256, 1, 1),
    "ag_news": (256, 1, 1)
}
def get_tokenizer_name(model_name=None):
    """
    根据模型名称返回对应的tokenizer名称
    Args:
        model_name: 模型名称，如 'pythia70m', 'roberta', 'opt13b' 等
    Returns:
        tokenizer名称
    """
    if model_name is None:
        # 默认使用 pythia-70m
        return "EleutherAI/pythia-70m"
    model_tokenizer_map = {
        "pythia70m": "EleutherAI/pythia-70m",
        "pythia70m_dropout": "EleutherAI/pythia-70m",
        "roberta": "roberta-base",
        "roberta_dropout": "roberta-base",
        "opt13b": "facebook/opt-1.3b",
        "opt13b_dropout": "facebook/opt-1.3b",
        "gpt2": "gpt2",
        "gpt2_dropout": "gpt2",
    }
    return model_tokenizer_map.get(model_name, "EleutherAI/pythia-70m")


def get_data(name,  model_name=None,augment=True, **kwargs):
    """
    加载数据集
    Args:
        name: 数据集名称
        augment: 是否使用数据增强（仅对图像数据集有效）
        model_name: 模型名称，用于选择对应的tokenizer（仅对文本数据集有效）
        **kwargs: 其他参数
    Returns:
        train_set, test_set
    """
    load_path=os.getcwd()+'/data'
    # 获取对应的tokenizer名称
    tokenizer_name = get_tokenizer_name(model_name)
    if name == "cifar10":
        if augment==True:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]

            train_set = datasets.CIFAR10(root=load_path, train=True,
                                         transform=transforms.Compose(train_transforms),
                                         download=True)

            test_set = datasets.CIFAR10(root=load_path, train=False,
                                        transform=transforms.Compose(
                                            train_transforms
                                        ), download=True)

        else:
            train_set = datasets.CIFAR10(root=load_path, train=True,transform=transforms.Compose([transforms.ToTensor()]),download=True)

            test_set = datasets.CIFAR10(root=load_path, train=False,transform=transforms.Compose([transforms.ToTensor()]), download=True)

    elif name == "fmnist":

        transform=transforms.ToTensor()

        train_set = datasets.FashionMNIST(root=load_path, train=True,
                                          transform=transform,
                                          download=True)

        test_set = datasets.FashionMNIST(root=load_path, train=False,
                                         transform=transform,
                                         download=True)

    elif name == "mnist":

        transform=transforms.ToTensor()
        train_set = datasets.MNIST(root=load_path, train=True,
                                   transform=transform,
                                   download=True)

        test_set = datasets.MNIST(root=load_path, train=False,
                                  transform=transform,
                                  download=True)

    elif name == "cifar100":

        if augment==True:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])

            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]


            train_set = datasets.CIFAR100(root=load_path, train=True,
                                          transform=transforms.Compose(train_transforms),
                                          download=True)

            test_set = datasets.CIFAR100(root=load_path, train=False,
                                         transform=transforms.Compose(
                                             [transforms.ToTensor(), normalize]
                                             ), download=True )
        else:

            train_set = datasets.CIFAR100(root=load_path, train=True,transform=transforms.Compose([transforms.ToTensor()]),
                                          download=True)

            test_set = datasets.CIFAR100(root=load_path, train=False,transform=transforms.Compose([transforms.ToTensor()]),
                                          download=True)
    elif name == "cinic10":
        if augment==True:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
            train_transforms =  transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ])


           # raw_train_set, raw_test_set = fetch_cinic10(load_path)  # need to download first

            train_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/train', transform=train_transforms)
            test_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/test', transform=train_transforms)
        else:
            train_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/train',transform=transforms.Compose([transforms.ToTensor()]))
            test_set = datasets.ImageFolder(root=f'{load_path}/cinic-10-batches-py/test',transform=transforms.Compose([transforms.ToTensor()]))

    elif name == "tinyimagenet":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_transforms =  transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])


      #  raw_train_set,raw_test_set = fetch_tinyimagenet(load_path)  # need to download first
        train_set = datasets.ImageFolder(root=f'{load_path}/tiny-imagenet-200/train', transform=train_transforms)
        test_set = datasets.ImageFolder(root=f'{load_path}/tiny-imagenet-200/val', transform=train_transforms)



    elif name == "svhn":
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                         std=[0.198, 0.201, 0.197])

        if augment:
            train_transforms = [
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.SVHN(root=load_path, split='train',
                                  transform=transforms.Compose(train_transforms),
                                  download=True)

        test_set = datasets.SVHN(root=load_path, split='test',
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), normalize]
                                 ), download=True)


    elif name == "celebA":  #need torchvision >0.17.0
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        augment = True

        if augment:
            train_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            train_transforms = [
                transforms.CenterCrop(178),
                transforms.Resize(128),
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CelebA(root=load_path, split='train', transform=transforms.Compose(train_transforms),
                                    download=True)
        test_set = datasets.CelebA(root=load_path, split='test', transform=transforms.Compose(
            [transforms.CenterCrop(178), transforms.Resize(128), transforms.ToTensor(), normalize]), download=True)

    elif name == "stl10":
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.198, 0.201, 0.197])


        augment = True
        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to 32x32
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.SVHN(root=load_path, split='train',
                                  transform=transforms.Compose(train_transforms),
                                  download=True)

        test_set = datasets.SVHN(root=load_path, split='test',
                                 transform=transforms.Compose(
                                     [transforms.ToTensor(), normalize]
                                 ), download=True)

    elif name == "sst5":
        try:
            dataset = load_dataset("SetFit/sst5", cache_dir=load_path)
            print("SST5数据集下载成功!")
            print(f"原始训练集大小: {len(dataset['train'])}")
            print(f"原始验证集大小: {len(dataset['validation'])}")
            print(f"原始测试集大小: {len(dataset['test'])}")
            
            def filter_valid(example):
                return 1 <= example['label'] <= 5
            
            # 过滤无效样本（label不在1-5范围内的样本）
            train_before_filter = len(dataset['train'])
            validation_before_filter = len(dataset['validation'])
            test_before_filter = len(dataset['test'])
            
            dataset['train'] = dataset['train'].filter(filter_valid)
            dataset['validation'] = dataset['validation'].filter(filter_valid)
            dataset['test'] = dataset['test'].filter(filter_valid)
            
            train_after_filter = len(dataset['train'])
            validation_after_filter = len(dataset['validation'])
            test_after_filter = len(dataset['test'])
            
            print(f"过滤后训练集大小: {train_after_filter} (过滤掉 {train_before_filter - train_after_filter} 个无效样本)")
            print(f"过滤后验证集大小: {validation_after_filter} (过滤掉 {validation_before_filter - validation_after_filter} 个无效样本)")
            print(f"过滤后测试集大小: {test_after_filter} (过滤掉 {test_before_filter - test_after_filter} 个无效样本)")
            
            from datasets import concatenate_datasets
            train_combined = concatenate_datasets([dataset['train'], dataset['validation']])
            
            print(f"合并后训练集大小: {len(train_combined)} (训练集 {train_after_filter} + 验证集 {validation_after_filter})")
            print(f"最终测试集大小: {len(dataset['test'])}")
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")
            
            train_set = SST5Dataset(train_combined, tokenizer, max_length=128)
            test_set = SST5Dataset(dataset['test'], tokenizer, max_length=128)
            
        except Exception as e:
            print(f"下载SST5数据集时出错: {e}")
            raise ValueError(f"无法加载SST5数据集: {e}")

    elif name == "news20":
        try:
            # 使用自定义缓存路径
            dataset = load_dataset("SetFit/20_newsgroups", cache_dir=load_path)
            print("20 Newsgroups数据集下载成功!")

            
            print(f"训练集大小: {len(dataset['train'])}")
            print(f"测试集大小: {len(dataset['test'])}")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")
            
            train_set = News20Dataset(dataset['train'], tokenizer, max_length=256)
            test_set = News20Dataset(dataset['test'], tokenizer, max_length=256)
            
        except Exception as e:
            print(f"下载20 Newsgroups数据集时出错: {e}")
            raise ValueError(f"无法加载20 Newsgroups数据集: {e}")

    elif name == "snli":
        try:
            dataset = load_dataset("snli", cache_dir=load_path)
            print("SNLI数据集下载成功!")

            def is_valid(example):
                return example['label'] != -1

            dataset['train'] = dataset['train'].filter(is_valid)
            dataset['validation'] = dataset['validation'].filter(is_valid)

            print(f"原始训练集大小: {len(dataset['train'])}")
            print(f"验证集大小: {len(dataset['validation'])}")

            # 从训练集中随机采样50000个样本
            train_size = len(dataset['train'])
            sample_size = 50000
            if train_size > sample_size:
                dataset['train'] = dataset['train'].shuffle(seed=42).select(range(sample_size))
                print(f"采样后训练集大小: {len(dataset['train'])}")
            else:
                print(f"训练集大小({train_size})小于50000，使用全部训练样本")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")

            train_set = PairedTextClassificationDataset(dataset['train'], tokenizer, text_field1='premise', text_field2='hypothesis', max_length=256)
            test_set = PairedTextClassificationDataset(dataset['validation'], tokenizer, text_field1='premise', text_field2='hypothesis', max_length=256)

        except Exception as e:
            print(f"下载SNLI数据集时出错: {e}")
            raise ValueError(f"无法加载SNLI数据集: {e}")

    elif name == "mnli":
        try:
            dataset = load_dataset("glue", "mnli", cache_dir=load_path)
            print("MNLI数据集下载成功!")

            print(f"原始训练集大小: {len(dataset['train'])}")
            print(f"验证集(匹配)大小: {len(dataset['validation_matched'])}")

            # 从训练集中随机采样50000个样本
            train_size = len(dataset['train'])
            sample_size = 50000
            if train_size > sample_size:
                dataset['train'] = dataset['train'].shuffle(seed=42).select(range(sample_size))
                print(f"采样后训练集大小: {len(dataset['train'])}")
            else:
                print(f"训练集大小({train_size})小于50000，使用全部训练样本")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")

            train_set = PairedTextClassificationDataset(dataset['train'], tokenizer, text_field1='premise', text_field2='hypothesis', max_length=256)
            test_set = PairedTextClassificationDataset(dataset['validation_matched'], tokenizer, text_field1='premise', text_field2='hypothesis', max_length=256)

        except Exception as e:
            print(f"下载MNLI数据集时出错: {e}")
            raise ValueError(f"无法加载MNLI数据集: {e}")
    elif name == "mrpc":
        try:
            dataset = load_dataset("glue", "mrpc", cache_dir=load_path)
            print("MRPC数据集下载成功!")
            print(f"训练集大小: {len(dataset['train'])}")
            print(f"验证集大小: {len(dataset['validation'])}")
            print(f"测试集大小: {len(dataset['test'])}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")

            train_set = PairedTextClassificationDataset(dataset['train'], tokenizer, text_field1='sentence1', text_field2='sentence2', max_length=256)
            test_set = PairedTextClassificationDataset(dataset['validation'], tokenizer, text_field1='sentence1', text_field2='sentence2', max_length=256)
        except Exception as e:
            print(f"下载MRPC数据集时出错: {e}")
            raise ValueError(f"无法加载MRPC数据集: {e}")
    elif name == "imdb":
        try:
            dataset = load_dataset("imdb", cache_dir=load_path)
            print("IMDB数据集下载成功!")
            print(f"训练集大小: {len(dataset['train'])}")
            print(f"测试集大小: {len(dataset['test'])}")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")
            train_set = TextClassificationDataset(dataset['train'], tokenizer, text_field='text', label_field='label', max_length=512)
            test_set = TextClassificationDataset(dataset['test'], tokenizer, text_field='text', label_field='label', max_length=512)

        except Exception as e:
            print(f"下载IMDB数据集时出错: {e}")
            raise ValueError(f"无法加载IMDB数据集: {e}")

    elif name == "rte":
        try:
            dataset = load_dataset("glue", "rte", cache_dir=load_path)
            print("RTE数据集下载成功!")
            print(f"训练集大小: {len(dataset['train'])}")
            print(f"验证集大小: {len(dataset['validation'])}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")

            train_set = PairedTextClassificationDataset(dataset['train'], tokenizer, text_field1='sentence1', text_field2='sentence2', max_length=256)
            test_set = PairedTextClassificationDataset(dataset['validation'], tokenizer, text_field1='sentence1', text_field2='sentence2', max_length=256)

        except Exception as e:
            print(f"下载RTE数据集时出错: {e}")
            raise ValueError(f"无法加载RTE数据集: {e}")

    elif name == "ag_news":
        try:
            dataset = load_dataset("ag_news", cache_dir=load_path)
            print("AG's News数据集下载成功!")
            print(f"训练集大小: {len(dataset['train'])}")
            print(f"测试集大小: {len(dataset['test'])}")

            # 检查标签范围
            train_labels = set([example['label'] for example in dataset['train']])
            test_labels = set([example['label'] for example in dataset['test']])
            all_labels = train_labels.union(test_labels)
            min_label = min(all_labels)
            max_label = max(all_labels)
            print(f"原始标签范围: {min_label} - {max_label}, 标签集合: {sorted(all_labels)}")

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=load_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"使用tokenizer: {tokenizer_name}")

            # 如果标签从1开始（1-4），转换为从0开始（0-3）
            # 如果标签已经从0开始（0-3），则不需要转换
            if min_label == 1 and max_label == 4:
                print("检测到标签从1开始，转换为从0开始")
                def adjust_label(example):
                    example['label'] = example['label'] - 1
                    return example
                dataset['train'] = dataset['train'].map(adjust_label)
                dataset['test'] = dataset['test'].map(adjust_label)
            elif min_label == 0 and max_label == 3:
                print("标签已经从0开始，无需转换")
            else:
                print(f"警告: 标签范围异常 {min_label}-{max_label}，期望 1-4 或 0-3")

            # 验证转换后的标签
            train_labels_after = set([example['label'] for example in dataset['train']])
            test_labels_after = set([example['label'] for example in dataset['test']])
            all_labels_after = train_labels_after.union(test_labels_after)
            print(f"转换后标签范围: {min(all_labels_after)} - {max(all_labels_after)}, 标签集合: {sorted(all_labels_after)}")
            
            # 确保标签在0-3范围内
            assert min(all_labels_after) >= 0 and max(all_labels_after) <= 3, \
                f"标签超出范围 [0-3]: {sorted(all_labels_after)}"

            train_set = TextClassificationDataset(dataset['train'], tokenizer, text_field='text', label_field='label', max_length=256)
            test_set = TextClassificationDataset(dataset['test'], tokenizer, text_field='text', label_field='label', max_length=256)

        except Exception as e:
            print(f"下载AG's News数据集时出错: {e}")
            raise ValueError(f"无法加载AG's News数据集: {e}")

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


