import torch.utils.data as data
from typing import Optional
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2


class DegradeDataset(data.Dataset):
    def __init__(
            self,
            img_dir: str,
            file_list: list[str],
            patch_size: int = 128,
            scale_factor: int = 2,
            noise_level: Optional[int] = 15,
            use_augm: bool = True
    ) -> None:
        self.img_dir = img_dir
        self.file_list = file_list
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        self.patch_size = patch_size
        self.use_augm = use_augm

    def degrade_image(self, img: np.ndarray, add_noise: bool, noise: int = 0) -> np.ndarray:
        height, width = img.shape[:2]

        img_lr = cv2.resize(img, (width // self.scale_factor, height // self.scale_factor), interpolation=cv2.INTER_CUBIC)

        if add_noise:
            if noise is None:
                noise = np.random.randint(5, 51)
            img_lr = img_lr.astype(np.float32)
            noise = np.random.normal(loc=0, scale=noise, size=img_lr.shape)
            img_lr = img_lr + noise
            img_lr = np.clip(img_lr, 0, 255)

            img_lr = img_lr.astype(np.uint8)

        return img_lr


    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int) -> np.ndarray:
        img_path = self.img_dir+self.file_list[index]
        img_hr = cv2.imread(img_path)
        if img_hr is None:
            raise FileNotFoundError
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        h, w = img_hr.shape[:2] 
        if self.patch_size is not None:
            if h >= self.patch_size and w >= self.patch_size:
                i = np.random.randint(0, h - self.patch_size)
                j = np.random.randint(0, w - self.patch_size)
                img_hr = img_hr[i:i+self.patch_size, j:j+self.patch_size]
        
        if np.random.random() > 0.5 and self.use_augm:
            img_hr = cv2.flip(img_hr, 1)

        return img_hr

class DenoiseDegradeDataset(DegradeDataset):
    def __init__(
            self, 
            img_dir: str,
            file_list: list[str],
            patch_size: int = 128,
            scale_factor: int = 2,
            noise_level: Optional[int] = 15,
            use_augm: bool = True,     
            use_noise_map: bool = False, 
            estimate_model = None
    ) -> None:
        super().__init__(img_dir, file_list, patch_size, scale_factor, noise_level, use_augm)
        self.use_noise_map = use_noise_map
        self.estimate_model = estimate_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_noise_map(self, img: np.ndarray, noise: int):
        if self.noise_level is None:
            self.estimate_model.eval()
            img = self.to_tensor(img).to(self.device)
            noise_map = self.estimate_model(img).cpu()
        else:
            noise_map = np.full((img.shape[0], img.shape[1]), noise / 255, dtype=np.float32)
            noise_map = self.to_tensor(noise_map)
        return noise_map

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_hr = super().__getitem__(index)
        # if self.noise_level is None:
        #     noise = np.random.randint(5, 51)
        # else:
        #     noise = self.noise_level
        noisy_lr = self.degrade_image(img_hr, add_noise=True, noise=self.noise_level)
        
        clean_lr = self.degrade_image(img_hr, add_noise=False)

        self.to_tensor = transforms.ToTensor()

        noisy_lr_t = self.to_tensor(noisy_lr)
        clean_lr_t = self.to_tensor(clean_lr)
        
        if self.use_noise_map:
            noise_map_t = self.get_noise_map(noisy_lr, self.noise_level)
            noisy_lr_t = torch.cat([noisy_lr_t, noise_map_t], dim=0)

        return (noisy_lr_t, clean_lr_t)


class SuperresDegradeDataset(DegradeDataset):
    def __init__(
            self, 
            img_dir: str,
            file_list: list[str],
            patch_size: int = 128,
            scale_factor: int = 2,
            noise_level: int = 15,
            use_augm: bool = True
    ) -> None:
        super().__init__(img_dir, file_list, patch_size, scale_factor, noise_level, use_augm)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_hr = super().__getitem__(index)
        img_lr = self.degrade_image(img_hr, add_noise=False)

        to_tensor = transforms.ToTensor()

        return (to_tensor(img_lr), to_tensor(img_hr))
    
class CascadeDataset(DegradeDataset):
    def __init__(
            self, 
            img_dir: str,
            file_list: list[str],
            patch_size: int = 128,
            scale_factor: int = 2,
            noise_level: int = 15,
            use_augm: bool = True
    ) -> None:
        super().__init__(img_dir, file_list, patch_size, scale_factor, noise_level, use_augm)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_hr = super().__getitem__(index)
        img_lr_noisy = self.degrade_image(img_hr, add_noise=True, noise=self.noise_level)

        to_tensor = transforms.ToTensor()

        return (to_tensor(img_lr_noisy), to_tensor(img_hr))
    
class NoiseEstimateDataset(DegradeDataset):
    def __init__(
            self, 
            img_dir: str,
            file_list: list[str],
            patch_size: int = 128,
            scale_factor: int = 2,
            noise_level: Optional[int] = 15,
            use_augm: bool = True, 
    ) -> None:
        super().__init__(img_dir, file_list, patch_size, scale_factor, noise_level, use_augm)
    
    def __getitem__(self, index):
        img_hr = super().__getitem__(index)
        noise = np.random.randint(5, 51)
        noisy_lr = self.degrade_image(img_hr, add_noise=True, noise=noise)

        to_tensor = transforms.ToTensor()

        noisy_lr_t = to_tensor(noisy_lr)
        
        target_map = np.full((noisy_lr.shape[0], noisy_lr.shape[1]), noise / 255, dtype=np.float32)
        target_tensor = to_tensor(target_map)
        
        return (noisy_lr_t, target_tensor)
        