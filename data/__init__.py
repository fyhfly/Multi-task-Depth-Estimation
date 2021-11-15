import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from data.datasets import get_dataset, ConcatDataset
from data.transform import RandomImgAugment, SegToTensor

def create_test_dataloader(args):

    joint_transform_list = [RandomImgAugment(args.loadSize)]
    img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]

    joint_transform = Compose(joint_transform_list)
    
    img_transform = Compose(img_transform_list)
    
    Seg_transform = Compose([SegToTensor()])

    dataset = get_dataset(root=args.root, data_file=args.test_datafile, phase='test',
                        dataset=args.dataset, img_transform=img_transform, Seg_transform=None,
                        joint_transform=joint_transform)
    loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=1, shuffle=False,
                                num_workers=int(args.nThreads),
                                pin_memory=True)
    
    return loader

def create_train_dataloader(args):
    joint_transform_list = [RandomImgAugment(args.loadSize)]
    img_transform_list = [ToTensor(), Normalize([.5, .5, .5], [.5, .5, .5])]

    joint_transform = Compose(joint_transform_list)
    
    img_transform = Compose(img_transform_list)

    Seg_transform = Compose([SegToTensor()])

    dataset = get_dataset(root=args.root, data_file=args.train_datafile, phase='train',
                            dataset=args.dataset,
                            img_transform=img_transform, joint_transform=joint_transform,
                            Seg_transform=Seg_transform)

    loader = torch.utils.data.DataLoader(
                                dataset,
                                batch_size=args.batchSize, shuffle=True,
                                num_workers=int(args.nThreads),
                                pin_memory=True)

    return loader


def create_dataloader(args):

    if not args.isTrain:
        return create_test_dataloader(args)

    else:
        return create_train_dataloader(args)
   