import os
import random
from glob import glob
from pdb import set_trace as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from utils.utils_model import *

class CADDataLoader(Dataset):
    def __init__(self, split='train', do_norm=True, cfg=None, max_prim=12000):
        self.set_random_seed(123)
        self.root = cfg.root
        self.split = split
        self.max_prim = max_prim
        self.debug = cfg.debug
        if cfg is not None:
            self.clus_num_per_batch = cfg.clus_num_per_batch
            self.nn = cfg.clus_nn
            self.size = cfg.img_size
            self.filter_num = cfg.filter_num
            self.aug_ratio = cfg.aug_ratio
            self.rgb_dim = cfg.rgb_dim
        else:
            self.clus_num_per_batch = 16
            self.nn = 64
            self.size = 700
            self.filter_num = 64
            self.aug_ratio = 0.5
            self.rgb_dim = 0
        # transformations
        transform = [T.ToTensor()]
        if do_norm:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        # pre-loading
        self.image_path_list = glob(os.path.join(self.root, "png", split, "*.png"))
        self.anno_path_list = glob(os.path.join(self.root, "npy", split, "*.npy"))
        self.image_path_list = sorted(self.image_path_list)
        self.anno_path_list = sorted(self.anno_path_list)

        # data augmentation
        self.train_len = len(self.anno_path_list)
        if ("train" in split) and (self.aug_ratio >= 1e-4):
            print(f" > before aug training: {len(self.anno_path_list)}")
            self.aug_training()
            print(f" > after aug training: {len(self.anno_path_list)}")


        if not self.debug:
            assert len(self.image_path_list) == len(self.anno_path_list)
        self.length = len(self.image_path_list)

        print(" > before filter_smallset:", len(self.anno_path_list))
        if not self.debug:
            self.image_path_list, self.anno_path_list = self.filter_smallset()

        if self.debug:
            if split == 'train':
                self.image_path_list, self.anno_path_list = self.image_path_list[:200], self.anno_path_list[:200]
            else:
                self.image_path_list, self.anno_path_list = self.image_path_list[:20], self.anno_path_list[:20]

        self.length = len(self.image_path_list)
        print(" > after filter_smallset:", len(self.anno_path_list))

    def filter_smallset(self):
        anno_path_list_new = []
        image_path_list_new = []
        for idx, ann_path in enumerate(self.anno_path_list):
            adj_node_classes = np.load(ann_path, \
                                    allow_pickle=True).item()
            target = adj_node_classes["cat"]
            if self.split == "training":
                if self.max_prim >= len(target) >= self.filter_num:
                    anno_path_list_new.append(self.anno_path_list[idx])
                    image_path_list_new.append(self.image_path_list[idx])
            else:
                if len(target) >= self.filter_num:
                    anno_path_list_new.append(self.anno_path_list[idx])
                    image_path_list_new.append(self.image_path_list[idx])
        return image_path_list_new, anno_path_list_new


    def __len__(self):
        return self.length


    def _get_item(self, index):
        img_path = self.image_path_list[index]
        ann_path = self.anno_path_list[index]
        assert os.path.basename(img_path).split(".")[0] == \
            os.path.basename(ann_path).split(".")[0]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.size, self.size))
        image = self.transform(image).cuda()

        adj_node_classes = np.load(ann_path, \
                            allow_pickle=True).item()
        target = adj_node_classes["cat"]
        target = torch.from_numpy(np.array(target, dtype=np.long)).cuda()

        center = adj_node_classes["ct_norm"]
        xy = torch.from_numpy(np.array(center, dtype=np.float32)).cuda()

        if self.rgb_dim > 0:
            rgb_npy_path = ann_path.replace('/npy/', '/npy_rgb/')
            rgb_info = np.load(rgb_npy_path, allow_pickle=True).item()['rgbs']
            rgb_info = torch.from_numpy(np.array(rgb_info, dtype=np.long)).cuda()
        else:
            rgb_info = xy

        nns = adj_node_classes["nns"]
        nns = torch.from_numpy(np.array(nns, dtype=np.long)).cuda()

        instance = adj_node_classes["inst"]
        instance_center = self.get_instance_center_tensor(instance, center, semantic=target, img_path=img_path)
        instance = torch.from_numpy(np.array(instance, dtype=np.float32)).cuda()
        offset = xy - instance_center

        indexes = torch.Tensor([1]).cuda()
        basename = os.path.basename(img_path)

        return image, xy, target, rgb_info, nns, offset, instance, indexes, basename


    def __getitem__(self, index):
        return self._get_item(index)


    def random_sample(self, image, xy, target, rgb_info, nns, offset, instance, indexes, basename):
        length = xy.shape[0]
        rand_idx = random.sample(range(length), self.max_prim)
        rand_idx = sorted(rand_idx)
        xy = xy[rand_idx]
        target = target[rand_idx]
        rgb_info = rgb_info[rand_idx]
        nns = nns[rand_idx]
        offset = offset[rand_idx]
        instance = instance[rand_idx]
        return image, xy, target, rgb_info, nns, offset, instance, indexes, basename


    def set_random_seed(self, seed, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def aug_training(self):
        self.image_path_list_aux = glob(os.path.join(self.root, "images", "{}_aug5x".format(self.split), "images", "*.png"))
        self.anno_path_list_aux = glob(os.path.join(self.root, "annotations", "{}_aug5x".format(self.split), "constructed_graphs_withnninst", "*.npy"))
        self.image_path_list_aux = sorted(self.image_path_list_aux)
        self.anno_path_list_aux = sorted(self.anno_path_list_aux)
        try:
            assert len(self.image_path_list_aux) == len(self.anno_path_list_aux)
        except:
            def extra_same_elem(list1, list2):
                set1 = set(list1)
                set2 = set(list2)
                iset = set1.intersection(set2)
                return list(iset)
            img_list = [os.path.basename(x).split(".")[0] for x in self.image_path_list_aux]
            ann_list = [os.path.basename(x).split(".")[0] for x in self.anno_path_list_aux]
            intersect = extra_same_elem(img_list, ann_list)
            img_dir = os.path.dirname(self.image_path_list_aux[0])
            ann_dir = os.path.dirname(self.anno_path_list_aux[0])
            self.image_path_list_aux = [os.path.join(img_dir, "{}.png".format(x)) for x in intersect]
            self.anno_path_list_aux = [os.path.join(ann_dir, "{}.npy".format(x)) for x in intersect]
            assert len(self.image_path_list_aux) == len(self.anno_path_list_aux)

        aux_len = len(self.anno_path_list_aux)
        aug_n = int(self.aug_ratio * self.train_len)
        aug_n = min(aug_n, aux_len)
        idxes = random.sample(range(0, aux_len-1), aug_n)
        self.image_path_list_aux = [self.image_path_list_aux[i] for i in idxes]
        self.anno_path_list_aux = [self.anno_path_list_aux[i] for i in idxes]
        self.image_path_list.extend(self.image_path_list_aux)
        self.anno_path_list.extend(self.anno_path_list_aux)


    def get_instance_center_tensor(self, instance, center, semantic=None, img_path=None):
        offset_list = []
        offset_dict = {}
        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val == -1:
                continue
            if inst_val in offset_dict.keys():
                offset_dict[inst_val]["cent"].append(center[idx])
            else:
                offset_dict[inst_val] = {}
                offset_dict[inst_val]["mean"] = None
                offset_dict[inst_val]["cent"] = []
                offset_dict[inst_val]["cent"].append(center[idx])

        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val != -1:
                offset_dict[inst_val]["mean"] = np.mean(offset_dict[inst_val]["cent"], axis=0)

        for idx, inst_num in enumerate(instance):
            inst_val = inst_num[0]
            if inst_val is None:
                st()
            if inst_val == -1 or inst_val is None:
                offset_list.append([-999, -999])
            else:
                try:
                    offset_list.append([offset_dict[inst_val]["mean"][0], offset_dict[inst_val]["mean"][1]])
                except:
                    st()
        instance_center = torch.from_numpy(np.array(offset_list, dtype=np.float32)).cuda()
        return instance_center

if __name__ == '__main__':
    pass
