import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR10

class SimCLRDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, _ = self.base_dataset[index]
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj
    
    def __len__(self):
        return len(self.base_dataset)
    
class SimCLRDataModule(LightningDataModule):
    def __init__(
            self,
            config
    ):
        super().__init__()

        self.data_dir = config["dataset"]["path"]
        self.train_batch_size = config["training"]["train_batch_size"]
        self.val_batch_size = config["training"]["val_batch_size"]
        self.num_workers = config["training"]["num_workers"]
        self.pin_memory = config["training"]["pin_memory"]

        # CONFIGS
        self.crop_size = config["augmentation"]["crop_size"] # cifar image size
        self.random_resized_crop_scale = config["augmentation"]["random_resized_crop"]["scale"]
        self.color_jitter_p = config["augmentation"]["color_jitter"]["p"]
        self.color_jitter_strength = config["augmentation"]["color_jitter"]["strength"]
        self.grayscale_prob = config["augmentation"]["grayscale_prob"]
        self.gaussian_blur_prob = config["augmentation"]["gaussian_blur_prob"]
        self.gaussian_blur_kernel_size = config["augmentation"]["gaussian_blur_kernel_size"]
        self.normalize_mean = config["augmentation"]["normalize"]["mean"]
        self.normalize_std = config["augmentation"]["normalize"]["std"]
    
    def setup(self, stage):
        
        color_jitter = transforms.ColorJitter(*self.color_jitter_strength)
        simclr_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.crop_size, scale=self.random_resized_crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=self.color_jitter_p),
            transforms.RandomGrayscale(p=self.grayscale_prob),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean,
                                std=self.normalize_std)
        ])



        base_train = CIFAR10(root=self.data_dir, train=True, download=True)
        base_val = CIFAR10(root=self.data_dir, train=False, download=True)

        self.train_dataset = SimCLRDataset(base_train, transform=simclr_transform)
        self.val_dataset = SimCLRDataset(base_val, transform=simclr_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                           batch_size=self.train_batch_size,
                           shuffle=True,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory,
                           drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                           batch_size=self.val_batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory=self.pin_memory,
                           drop_last=False)

