import torch.utils.data
import torchvision

from .ytvos import build as build_ytvos
from .davis import build as build_davis
from .refexp import build as build_refexp

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ytvos':
        print("\n **** Start to build dataset {}. **** \n".format("build_ytvos"))
        return build_ytvos(image_set, args)

    # for pretraining
    if dataset_file == "refcoco" or dataset_file == "refcoco+" or dataset_file == "refcocog":
        print("\n **** Start to build dataset {}. **** \n".format("build_refexp"))
        return build_refexp(dataset_file, image_set, args)

    raise ValueError(f'dataset {dataset_file} not supported')
