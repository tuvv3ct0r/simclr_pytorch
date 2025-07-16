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
            data_path,
            train_batch_size,
            val_batch_size,
            image_size,
            num_workers,
            pin_memory
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage):
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        simclr_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
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

