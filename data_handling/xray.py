from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from skimage import io
from torchvision.transforms import ToTensor, Resize, CenterCrop
from data_handling.base import BaseDataModuleClass
from datetime import datetime
import os
from data_handling.caching import SharedCache
from data_handling.augmentations import DataAugmentationDINO

# Please update this with your own paths.
DATA_DIR_RSNA = Path("/vol/biomedic3/mb121/rsna-pneumonia-detection-challenge")
DATA_DIR_RSNA_PROCESSED_IMAGES = DATA_DIR_RSNA / "preprocess_224_224"
PATH_TO_PNEUMONIA_WITH_METADATA_CSV = (
    Path(__file__).parent / "pneumonia_dataset_with_metadata.csv"
)

cluster_root = os.getenv("HOME", None)

if Path("/data2/PadChest").exists():
    PADCHEST_ROOT = Path("/data2/PadChest")
    PADCHEST_IMAGES = PADCHEST_ROOT / "preprocessed"
elif (Path(cluster_root) / "PadChest").exists():
    PADCHEST_ROOT = Path(cluster_root) / "PadChest"
    PADCHEST_IMAGES = PADCHEST_ROOT / "preprocessed"
else:
    PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
    PADCHEST_IMAGES = PADCHEST_ROOT / "images"


def prepare_padchest_csv():
    df = pd.read_csv(
        PADCHEST_ROOT / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    )
    df = df.loc[df.Projection == "PA"]
    df = df.loc[df.Pediatric == "No"]

    def process(x, target):
        if isinstance(x, str):
            list_labels = x[1:-1].split(",")
            list_labels = [label.replace("'", "").strip() for label in list_labels]
            return target in list_labels
        else:
            return False

    for label in [
        "pneumonia",
        "exclude",
        "suboptimal study",
    ]:
        df[label] = df.Labels.astype(str).apply(lambda x: process(x, label))
        print(df[label].value_counts())
    df = df.loc[~df.exclude]
    df = df.loc[~df["suboptimal study"]]
    df["Manufacturer"] = df.Manufacturer_DICOM.apply(
        lambda x: "Phillips" if x == "PhilipsMedicalSystems" else "Imaging"
    )
    df = df.loc[df["PatientSex_DICOM"].isin(["M", "F"])]
    df["PatientAge"] = (
        df.StudyDate_DICOM.apply(lambda x: datetime.strptime(str(x), "%Y%M%d").year)
        - df.PatientBirth
    )
    invalid_filenames = [
        "216840111366964013829543166512013353113303615_02-092-190.png",
        "216840111366964013962490064942014134093945580_01-178-104.png",
        "216840111366964012989926673512011151082430686_00-157-045.png",
        "216840111366964012558082906712009327122220177_00-102-064.png",
        "216840111366964012959786098432011033083840143_00-176-115.png",
        "216840111366964012373310883942009152114636712_00-102-045.png",
        "216840111366964012487858717522009280135853083_00-075-001.png",
        "216840111366964012819207061112010307142602253_04-014-084.png",
        "216840111366964012989926673512011074122523403_00-163-058.png",
        "216840111366964013590140476722013058110301622_02-056-111.png",
        "216840111366964012339356563862009072111404053_00-043-192.png",
        "216840111366964013590140476722013043111952381_02-065-198.png",
        "216840111366964012819207061112010281134410801_00-129-131.png",
        "216840111366964013686042548532013208193054515_02-026-007.png",
        "216840111366964012989926673512011083134050913_00-168-009.png"
        # '216840111366964013590140476722013058110301622_02-056-111.png'
    ]
    df = df.loc[~df.ImageID.isin(invalid_filenames)]
    return df


class PadChestDataModule(BaseDataModuleClass):
    def create_datasets(self):
        df = prepare_padchest_csv()
        label_col = self.config.data.label

        train_val_id, test_id = train_test_split(
            df.PatientID.unique(),
            test_size=0.20,
            random_state=33,
        )

        train_id, val_id = train_test_split(
            train_val_id,
            test_size=0.10,
            random_state=33,
        )

        if self.config.data.prop_train < 1.0:
            rng = np.random.default_rng(33)
            rng = np.random.default_rng(self.config.seed)
            train_id = rng.choice(
                train_id,
                size=int(self.config.data.prop_train * train_id.shape[0]),
                replace=False,
            )

        self.dataset_train = PadChestDataset(
            df=df.loc[df.PatientID.isin(train_id)],
            transform=self.train_tsfm,
            label_column=label_col,
            parents=self.parents,
            cache=self.config.data.cache,
            use_counterfactuals=self.config.data.use_counterfactuals,
            counterfactual_contrastive_pairs=self.config.data.counterfactual_contrastive,
        )

        self.dataset_val = PadChestDataset(
            df=df.loc[df.PatientID.isin(val_id)],
            transform=self.val_tsfm,
            label_column=label_col,
            parents=self.parents,
            cache=self.config.data.cache,
        )

        self.dataset_test = PadChestDataset(
            df=df.loc[df.PatientID.isin(test_id)],
            transform=self.val_tsfm,
            label_column=label_col,
            cache=True,
        )

    @property
    def dataset_name(self):
        return "padchest"

    @property
    def num_classes(self):
        return 2


class CheXpertDataModule(BaseDataModuleClass):
    def create_datasets(self):
        label_col = self.config.data.label
        df = pd.read_csv("/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/train.csv")
        df.fillna(0, inplace=True)  # assume no mention is like negative
        df = df.loc[df["AP/PA"] == "PA"]
        df = df.loc[df[self.config.data.label] != -1]  # remove the uncertain cases
        df["PatientID"] = df["Path"].apply(
            lambda x: int(Path(x).parent.parent.stem[-5:])
        )
        patient_id = df["PatientID"].unique()
        train_val_id, test_id = train_test_split(
            patient_id, test_size=0.4, random_state=33
        )
        train_id, val_id = train_test_split(
            train_val_id, test_size=0.15, random_state=33
        )

        if self.config.data.prop_train < 1.0:
            rng = np.random.default_rng(self.config.seed)
            train_id = rng.choice(
                train_id,
                size=int(self.config.data.prop_train * train_id.shape[0]),
                replace=False,
            )

        self.dataset_train = CheXpertDataset(
            df=df.loc[df.PatientID.isin(train_id)],
            transform=self.train_tsfm,
            cache=self.config.data.cache,
            label_col=label_col,
        )

        self.dataset_val = CheXpertDataset(
            df=df.loc[df.PatientID.isin(val_id)],
            transform=self.val_tsfm,
            cache=self.config.data.cache,
            label_col=label_col,
        )

        self.dataset_test = CheXpertDataset(
            df=df.loc[df.PatientID.isin(test_id)],
            transform=self.val_tsfm,
            cache=True,
            label_col=label_col,
        )

    @property
    def dataset_name(self):
        return "chexpert"

    @property
    def num_classes(self):
        return 2


class PadChestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_column: str,
        transform: Callable,
        parents: Optional = None,
        cache: bool = False,
        use_counterfactuals: bool = False,
        counterfactual_contrastive_pairs: bool = True,
    ):
        super().__init__()
        print(f"Len {len(df)}")
        self.counterfactual_contrastive_pairs = counterfactual_contrastive_pairs
        self.parents = parents
        self.use_counterfactuals = use_counterfactuals
        self.label_col = label_column
        self.pneumonia = df.pneumonia.astype(int).values
        self.img_paths = df.ImageID.values
        self.genders = df.PatientSex_DICOM.values
        self.ages = df.PatientAge.values
        self.manufacturers = df.Manufacturer.values
        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        try:
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
        except:  # noqa
            from PIL import ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            print(self.img_paths[idx])
            img = io.imread(PADCHEST_IMAGES / self.img_paths[idx], as_gray=True)
            print("success")
            ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img.float()

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=True)

        else:
            img = self.read_image(idx)

        sample = {}
        sample["pneumonia"] = self.pneumonia[idx]
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.genders[idx] == "M" else 1
        sample["scanner"] = 0 if self.manufacturers[idx] == "Phillips" else 1
        sample["y"] = sample[self.label_col]
        sample["shortpath"] = self.img_paths[idx]

        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()

        if isinstance(self.transform, DataAugmentationDINO):
            if self.use_counterfactuals:
                if self.counterfactual_contrastive_pairs:
                    # Sample domain if same as real then use real
                    cfx = (
                        self.load_counterfactual_image(idx)
                        if torch.rand(1).item() > 0.5
                        else img.clone()
                    )
                else:
                    if torch.rand(1).item() > 0.5:
                        cfx = self.load_counterfactual_image(idx)
                        img = cfx.clone()
                    else:
                        cfx = img.clone()
                img = torch.stack([img, cfx], dim=0)
            img = self.transform(img.float())
            sample.update(img)
        else:
            if self.use_counterfactuals:
                if self.counterfactual_contrastive_pairs:
                    cfx = self.load_counterfactual_image(idx)
                else:
                    if torch.rand(1).item() > 0.5:
                        cfx = self.load_counterfactual_image(idx)
                        img = cfx.clone()
                    else:
                        cfx = img.clone()
                img = self.transform(img)
                cfx = self.transform(cfx)
                img = torch.stack([img, cfx], dim=0)
            else:
                img = self.transform(img)

            sample["x"] = img.float()

        return sample

    def load_counterfactual_image(self, idx):
        cf_dir = Path("/vol/biomedic3/mb121/causal-contrastive/padchest_cf_images_v0")
        short_path = self.img_paths[idx][:-4]
        filename = cf_dir / f"{short_path}_sc_cf.png"
        img = io.imread(str(filename), as_gray=True) / 255.0
        img = img / (img.max() + 1e-12)
        img = ToTensor()(img)
        assert img.max() <= 1
        return img


class CheXpertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        transform: Callable,
        cache: bool = False,
    ):
        super().__init__()
        print(f"Len dataset {len(df)}")
        df.fillna(0, inplace=True)
        self.labels = df[label_col].astype(int).values
        self.img_paths = df.Path.values
        self.genders = df.Sex.values
        self.ages = df.Age.values
        self.cache = cache
        self.transform = transform

        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=self.img_paths.shape[0],
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __len__(self):
        return len(self.img_paths)

    def read_image(self, idx):
        img = io.imread(
            Path("/vol/biodata/data/chest_xray") / self.img_paths[idx], as_gray=True
        )
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            img = self.cache.get_slot(idx)
            if img is None:
                img = self.read_image(idx)
                self.cache.set_slot(idx, img, allow_overwrite=True)
        else:
            img = self.read_image(idx)

        sample = {}
        sample["age"] = self.ages[idx] / 100
        sample["sex"] = 0 if self.genders[idx] == "Male" else 1
        sample["y"] = self.labels[idx]
        sample["shortpath"] = self.img_paths[idx]
        sample["x"] = self.transform(img).float()
        return sample


class RNSAPneumoniaDetectionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        parents: Optional = None,
        cache: bool = False,
        use_counterfactuals: bool = False,
        counterfactual_contrastive_pairs: bool = True,
    ) -> None:
        """
        Torchvision dataset for loading RSNA dataset.
        Args:
            root: the data directory where the images can be found
            dataframe: the csv file mapping patient id, metadata, file names and label.
            transform: the transformation (i.e. preprocessing and / or augmentation)
            to apply to the image after loading them.

        This dataset returns a dictionary with the image data, label and metadata.
        """
        super().__init__()
        self.transform = transform
        self.parents = parents
        self.use_counterfactuals = use_counterfactuals
        self.counterfactual_contrastive_pairs = counterfactual_contrastive_pairs
        self.df = df
        self.targets = self.df.label_rsna_pneumonia.values.astype(np.int64)
        self.subject_ids = self.df.patientId.unique()
        self.filenames = [
            DATA_DIR_RSNA_PROCESSED_IMAGES / f"{subject_id}.png"
            for subject_id in self.subject_ids
        ]
        self.genders = self.df["Patient Gender"].values
        self.ages = self.df["Patient Age"].values.astype(int)
        if cache:
            self.cache = SharedCache(
                size_limit_gib=24,
                dataset_len=len(self.filenames),
                data_dims=[1, 224, 224],
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def read_image(self, idx):
        img = io.imread(self.filenames[idx], as_gray=True)
        img = img / (img.max() + 1e-12)
        img = CenterCrop(224)(Resize(224, antialias=True)(ToTensor()(img)))
        return img

    def __getitem__(self, index: int):
        img = self.read_image(index)
        sample = {
            "y": self.targets[index],
            "gender": self.genders[index],
            "pneumonia": self.targets[index],
            "sex": 1 if self.genders[index] == "M" else 0,
            "age": self.ages[index],
            "scanner": np.nan,
        }

        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()

        if self.use_counterfactuals:
            if torch.rand(1).item() > 0.5:
                cfx = img.clone()
            else:
                cfx = self.load_counterfactual_image(index)
                if not self.counterfactual_contrastive_pairs:
                    img = cfx.clone()
            img = self.transform(img)
            cfx = self.transform(cfx)
            img = torch.stack([img, cfx], dim=0).float()
        else:
            img = self.transform(img).float()

        sample["x"] = img

        return sample

    def __len__(self) -> int:
        return len(self.filenames)

    def load_counterfactual_image(self, index):
        raise NotImplementedError


class RSNAPneumoniaDataModule(BaseDataModuleClass):
    def create_datasets(self):
        """
        Pytorch Lightning DataModule defining train / val / test splits for the RSNA dataset.
        """
        if not DATA_DIR_RSNA_PROCESSED_IMAGES.exists():
            print(
                f"Data dir: {DATA_DIR_RSNA_PROCESSED_IMAGES} does not exist."
                + " Have you updated default_paths.py?"
            )

        if not PATH_TO_PNEUMONIA_WITH_METADATA_CSV.exists():
            print(
                """
                The dataset can be found at
                https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
                This dataset is originally a (relabelled) subset of the NIH dataset
                https://www.kaggle.com/datasets/nih-chest-xrays/data from
                which i took the metadata.
                To get the full csv with all the metadata please run
                data_handling/csv_generation_code/rsna_generate_full_csv.py
                """
            )
        df_with_all_labels = pd.read_csv(PATH_TO_PNEUMONIA_WITH_METADATA_CSV)
        df_with_all_labels = df_with_all_labels.loc[
            df_with_all_labels["View Position"] == "PA"
        ]

        random_seed_for_splits = 33

        indices_train_val, indices_test = train_test_split(
            np.arange(len(df_with_all_labels)),
            test_size=0.3,
            random_state=random_seed_for_splits,
        )
        train_val_df = df_with_all_labels.iloc[indices_train_val]
        test_df = df_with_all_labels.iloc[indices_test]

        # Further split train and val
        indices_train, indices_val = train_test_split(
            np.arange(len(train_val_df)),
            test_size=0.15,
            random_state=random_seed_for_splits,
        )

        if self.config.data.prop_train < 1.0:
            rng = np.random.default_rng(33)
            indices_train = rng.choice(
                indices_train,
                size=int(self.config.data.prop_train * indices_train.shape[0]),
                replace=False,
            )

        train_df = train_val_df.iloc[indices_train]
        val_df = train_val_df.iloc[indices_val]
        print(
            f"N patients train {indices_train.shape[0]}, val {indices_val.shape[0]}, test {indices_test.shape[0]}"  # noqa
        )

        self.dataset_train = RNSAPneumoniaDetectionDataset(
            df=train_df,
            transform=self.train_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
            use_counterfactuals=self.config.data.use_counterfactuals,
            counterfactual_contrastive_pairs=self.config.data.counterfactual_contrastive,
        )
        self.dataset_val = RNSAPneumoniaDetectionDataset(
            df=val_df,
            transform=self.val_tsfm,
            parents=self.parents,
            cache=self.config.data.cache,
        )

        self.dataset_test = RNSAPneumoniaDetectionDataset(
            df=test_df,
            transform=self.val_tsfm,
            parents=self.parents,
            cache=True,
        )

        print("#train: ", len(self.dataset_train))
        print("#val:   ", len(self.dataset_val))
        print("#test:  ", len(self.dataset_test))

    @property
    def num_classes(self):
        return 2
