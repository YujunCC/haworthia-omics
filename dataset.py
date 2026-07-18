import os
import random
import sqlite3
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from database import DB_PATH
from segmentation import segmentation_engine


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pad_to_square(image, fill):
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, (left, top))
    return canvas


def _split_rgb_and_alpha(image, alpha_image=None):
    if alpha_image is None:
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        source_rgb = rgba.convert("RGB")
    else:
        source_rgb = image.convert("RGB")
        alpha = alpha_image.convert("L")
        if alpha.size != source_rgb.size:
            alpha = alpha.resize(source_rgb.size, Image.Resampling.BILINEAR)

    rgb = Image.new("RGB", source_rgb.size, (0, 0, 0))
    rgb.paste(source_rgb, mask=alpha)
    return rgb, alpha


class PairedImageTransform:
    """Apply identical geometry to RGB and alpha while preserving the whole plant."""

    def __init__(self, input_size=224, training=False):
        self.input_size = input_size
        self.training = training
        self.color_jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15
        )

    def apply_with_mask(self, image, alpha_image=None):
        rgb, alpha = _split_rgb_and_alpha(image, alpha_image)
        rgb = _pad_to_square(rgb, (0, 0, 0))
        alpha = _pad_to_square(alpha, 0)

        rgb = TF.resize(
            rgb, [self.input_size, self.input_size], interpolation=InterpolationMode.BILINEAR
        )
        alpha = TF.resize(
            alpha, [self.input_size, self.input_size], interpolation=InterpolationMode.BILINEAR
        )

        if self.training:
            if random.random() < 0.5:
                rgb = TF.hflip(rgb)
                alpha = TF.hflip(alpha)

            angle = random.uniform(-20.0, 20.0)
            translate = [
                int(random.uniform(-0.04, 0.04) * self.input_size),
                int(random.uniform(-0.04, 0.04) * self.input_size),
            ]
            scale = random.uniform(0.88, 1.0)
            rgb = TF.affine(
                rgb, angle, translate, scale, 0.0,
                interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0)
            )
            alpha = TF.affine(
                alpha, angle, translate, scale, 0.0,
                interpolation=InterpolationMode.BILINEAR, fill=0
            )
            rgb = self.color_jitter(rgb)

        image_tensor = TF.normalize(TF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = TF.to_tensor(alpha).clamp(0.0, 1.0)
        return image_tensor, mask_tensor

    def __call__(self, image):
        image_tensor, _ = self.apply_with_mask(image)
        return image_tensor


train_transforms = PairedImageTransform(input_size=224, training=True)
val_transforms = PairedImageTransform(input_size=224, training=False)


class HaworthiaMultiViewDataset(Dataset):
    def __init__(
            self, db_path=DB_PATH, transform=None, is_training=False,
            quality_aware=False, minimum_quality_weight=0.35,
    ):
        self.transform = transform
        self.is_training = is_training
        self.quality_aware = quality_aware
        self.minimum_quality_weight = minimum_quality_weight
        self.samples = []

        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT tax_id, orig_path, seg_path FROM images")
        suspicious_count = 0
        quality_scores = []
        for tax_id, orig_path, seg_path in c.fetchall():
            if os.path.exists(orig_path) and os.path.exists(seg_path):
                quality_score = 1.0
                if quality_aware:
                    try:
                        with Image.open(seg_path) as segmented:
                            alpha = (
                                segmented.getchannel("A")
                                if "A" in segmented.getbands()
                                else Image.new("L", segmented.size, 255)
                            )
                            metrics = segmentation_engine.analyze_alpha(alpha, 70)
                            quality_score = metrics["quality_score"]
                            if metrics["suspicious"]:
                                quality_score *= 0.65
                                suspicious_count += 1
                    except Exception:
                        quality_score = 0.0
                        suspicious_count += 1
                quality_weight = minimum_quality_weight + (
                    (1.0 - minimum_quality_weight) * quality_score
                )
                quality_scores.append(quality_score)
                self.samples.append(
                    (orig_path, seg_path, tax_id, float(quality_weight))
                )
        conn.close()
        self.quality_summary = {
            "enabled": quality_aware,
            "suspicious_count": suspicious_count,
            "mean_score": (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            ),
            "minimum_weight": minimum_quality_weight,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        orig_path, seg_path, label, quality_weight = self.samples[idx]
        image = Image.open(orig_path).convert("RGB")
        with Image.open(seg_path) as segmented:
            if "A" in segmented.getbands():
                alpha = segmented.getchannel("A").copy()
            else:
                alpha = Image.new("L", segmented.size, 255)

        if self.transform is None:
            raise ValueError("A paired image transform is required.")

        if self.is_training:
            img_v1, mask_v1 = self.transform.apply_with_mask(image, alpha)
            img_v2, mask_v2 = self.transform.apply_with_mask(image, alpha)
            return img_v1, mask_v1, img_v2, mask_v2, label, quality_weight

        image_tensor, mask_tensor = self.transform.apply_with_mask(image, alpha)
        return image_tensor, mask_tensor, label


class PKBatchSampler(Sampler):
    def __init__(
            self, dataset, p_classes=16, k_instances=4,
            quality_sampling_strength=0.5,
    ):
        self.class_indices = defaultdict(list)
        self.sample_weights = {}
        for idx, (_, _, label, quality_weight) in enumerate(dataset.samples):
            self.class_indices[label].append(idx)
            self.sample_weights[idx] = (
                (1.0 - quality_sampling_strength)
                + quality_sampling_strength * quality_weight
            )
        self.classes = list(self.class_indices.keys())
        self.p_classes = min(p_classes, len(self.classes))
        self.k_instances = k_instances
        self.batch_size = self.p_classes * self.k_instances
        self.num_batches = max(1, len(dataset.samples) // self.batch_size)

    def _weighted_sample(self, indices):
        if len(indices) < self.k_instances:
            weights = [self.sample_weights[index] for index in indices]
            return random.choices(indices, weights=weights, k=self.k_instances)

        remaining = list(indices)
        selected = []
        for _ in range(self.k_instances):
            weights = [self.sample_weights[index] for index in remaining]
            chosen = random.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            sampled_classes = random.sample(self.classes, self.p_classes)
            for class_id in sampled_classes:
                indices = self.class_indices[class_id]
                batch.extend(self._weighted_sample(indices))
            yield batch

    def __len__(self):
        return self.num_batches
