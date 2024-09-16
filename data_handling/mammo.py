from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from torchvision.transforms import Resize
from data_handling.augmentations import DataAugmentationDINO

import os

cluster_root = os.getenv("HOME", None)

if Path("/data2/mb121/EMBED").exists():
    EMBED_ROOT = "/data2/mb121/EMBED"
if Path("/data/EMBED").exists():
    EMBED_ROOT = "/data/EMBED"
elif cluster_root is not None and (Path(cluster_root) / "EMBED").exists():
    EMBED_ROOT = Path(cluster_root) / "EMBED"
else:
    EMBED_ROOT = "/vol/biomedic3/data/EMBED"


VINDR_MAMMO_DIR = Path("/vol/biomedic3/data/VinDR-Mammo")

domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {"A": 0, "B": 1, "C": 2, "D": 3}
modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}


def preprocess_breast(image_path, target_size):
    """
    Loads the image performs basic background removal around the breast.
    Works for text but not for objects in contact with the breast (as it keeps the
    largest non background connected component.)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        # sometimes bug in reading images with cv2
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    # Connected components with stats.
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
        key=lambda x: x[1],
    )
    mask = output == max_label
    img = torch.tensor((gray * mask) / 255.0).unsqueeze(0).float()
    img = Resize(target_size, antialias=True)(img)
    return img


def get_embed_csv():
    image_dir = EMBED_ROOT / Path("images/png/1024x768")
    try:
        mydf = pd.read_csv(Path(__file__).parent / "joined_simple.csv")
    except FileNotFoundError:
        print(
            """
            For running EMBED code you need to first generate the csv
            file used for this study in csv_generation_code/generate_embed_csv.ipynb
            """
        )

    mydf["shortimgpath"] = mydf["image_path"]
    mydf["image_path"] = mydf["image_path"].apply(lambda x: image_dir / str(x))

    mydf["manufacturer_domain"] = mydf.Manufacturer.apply(lambda x: domain_maps[x])

    # convert tissueden to trainable label
    mydf["tissueden"] = mydf.tissueden.apply(lambda x: tissue_maps[x])

    mydf["SimpleModelLabel"] = mydf.ManufacturerModelName.apply(
        lambda x: modelname_map[x]
    )
    print(mydf.SimpleModelLabel.value_counts())
    mydf["ViewLabel"] = mydf.ViewPosition.apply(lambda x: 0 if x == "MLO" else 1)

    mydf["CviewLabel"] = mydf.FinalImageType.apply(lambda x: 0 if x == "2D" else 1)

    mydf = mydf.dropna(
        subset=[
            "age_at_study",
            "tissueden",
            "SimpleModelLabel",
            "ViewLabel",
            "image_path",
        ]
    )

    return mydf


def convert_vars_to_text(vars):
    sentence = ""
    if "scanner" in vars.keys():
        sentence += f"Scanner: {vars['scanner']}."
    if "cview" in vars.keys():
        sentence += " Type: FFDM." if vars["cview"] == 0 else " Type: Cview."
    if "view" in vars.keys():
        sentence += f" {vars['view']} view."
    if "age" in vars.keys():
        sentence += f" Patient age: {vars['age']}."
    if "density" in vars.keys():
        sentence += f" Density: {vars['density']}."
    return sentence


class EmbedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        parents,
        target_size,
        label="tissueden",
        cache: bool = True,
        use_counterfactuals: bool = False,
        counterfactual_contrastive_pairs: bool = True,
    ) -> None:
        self.imgs_paths = df.image_path.values
        self.shortpaths = df.shortimgpath.values
        self.labels = df[label].values
        print(df[label].value_counts())

        self.transform = transform
        self.target_size = target_size
        self.views = df.ViewLabel.values
        self.scanner = df.SimpleModelLabel.values
        self.cview = df.FinalImageType.apply(lambda x: 0 if x == "2D" else 1).values
        self.age = df.age_at_study.values
        self.parents = parents
        print(self.parents)
        self.densities = df.tissueden.values
        data_dims = [1, self.target_size[0], self.target_size[1]]
        if cache:
            self.cache = SharedCache(
                size_limit_gib=96,
                dataset_len=self.labels.shape[0],
                data_dims=data_dims,
                dtype=torch.float32,
            )
        else:
            self.cache = None
        self.use_counterfactuals = use_counterfactuals
        self.counterfactual_contrastive_pairs = counterfactual_contrastive_pairs

    def __getitem__(self, index) -> Any:
        if self.cache is not None:
            # retrieve data from cache if it's there
            img = self.cache.get_slot(index)
            # x will be None if the cache slot was empty or OOB
            if img is None:
                img = preprocess_breast(self.imgs_paths[index], self.target_size)
                self.cache.set_slot(index, img, allow_overwrite=True)  # try to cache x
        else:
            img = preprocess_breast(self.imgs_paths[index], self.target_size)
        sample = {}
        age = self.age[index]
        sample["cview"] = self.cview[index]
        sample["shortpath"] = str(self.shortpaths[index])
        sample["real_age"] = age
        sample["view"] = self.views[index]
        sample["density"] = torch.nn.functional.one_hot(
            torch.tensor(self.densities[index]).long(), num_classes=4
        ).detach()
        sample["y"] = self.labels[index]
        sample["scanner_int"] = self.scanner[index]
        sample["scanner"] = torch.nn.functional.one_hot(
            torch.tensor(self.scanner[index]).long(), num_classes=6
        ).detach()

        # Only used for causal models
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
                    if torch.rand(1).item() > 0.2:
                        cfx = self.load_counterfactual_image(index)
                    else:
                        cfx = img.clone()
                else:
                    if torch.rand(1).item() > 0.2:
                        cfx = self.load_counterfactual_image(index)
                        img = cfx.clone()
                    else:
                        cfx = img.clone()
                img = torch.stack([img, cfx], dim=0)
            img = self.transform(img.float())
            sample.update(img)
        else:
            if self.use_counterfactuals:
                if self.counterfactual_contrastive_pairs:
                    if torch.rand(1).item() > 0.2:
                        cfx = self.load_counterfactual_image(index)
                    else:
                        cfx = img.clone()
                else:
                    if torch.rand(1).item() > 0.2:
                        cfx = self.load_counterfactual_image(index)
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

    def __len__(self):
        return self.labels.shape[0]

    def load_counterfactual_image(self, index):
        short_path = str(self.shortpaths[index])[:-4]

        s = int(
            np.random.choice([a for a in range(5) if a != self.scanner[index]], size=1)
        )

        cf_dir1 = Path(
            "/vol/biomedic3/mb121/causal-contrastive/cf_beta1balanced_scanner"
        )

        filename = cf_dir1 / f"{short_path}_s{s}.png"

        img = io.imread(str(filename), as_gray=True) / 255.0
        assert img.max() <= 1

        return torch.tensor(img).unsqueeze(0).float()


class EmbedDataModule(BaseDataModuleClass):
    @property
    def dataset_name(self) -> str:
        return "EMBED"

    def get_all_csv_from_config(
        self, orig_df, label, domain=None, model=None, exclude_cviews=True
    ):
        df = orig_df.copy()
        print(len(df))
        if exclude_cviews:
            df = df.loc[df.FinalImageType == "2D"]
        print(len(df))
        if label == "tissueden":
            y = (
                df.groupby("empi_anon")["tissueden"]
                .unique()
                .apply(lambda x: x[0])
                .values
            )
            train_id, val_id = train_test_split(
                df.empi_anon.unique(), test_size=0.25, random_state=33, stratify=y
            )
        else:
            train_id, val_id = train_test_split(
                df.empi_anon.unique(), test_size=0.25, random_state=33
            )

        # 25% of patients - 600 for test
        test_id = val_id[600:]

        # 600 patients for val
        val_id = val_id[:600]

        print(
            f"N patients train: {train_id.shape[0]}, val: {val_id.shape[0]}, test {test_id.shape[0]}"
        )  # noqa

        if self.config.data.prop_train < 1.0:
            train_id = np.sort(train_id)
            y = (
                df.loc[df["empi_anon"].isin(train_id)]
                .groupby("empi_anon")["SimpleModelLabel"]
                .unique()
                .apply(lambda x: x[0])
                .sort_index()
            )
            assert y.index[0] == train_id[0]

            train_id, _ = train_test_split(
                train_id,
                train_size=int(self.config.data.prop_train * train_id.shape[0]),
                stratify=y.values,
                random_state=self.config.seed,
            )

        if domain != "None" and domain is not None:
            df = df.loc[df.manufacturer_domain == domain]
            if domain == domain_maps["HOLOGIC, Inc."]:
                df = df.loc[df.ManufacturerModelName == "Selenia Dimensions"]
        if model != "None" and model is not None:
            df = df.loc[df.SimpleModelLabel == model]

        print("train n images" + str(len(df.loc[df.empi_anon.isin(train_id)])))
        print("test" + str(len(df.loc[df.empi_anon.isin(test_id)])))
        print("val" + str(len(df.loc[df.empi_anon.isin(val_id)])))
        print("Scanner in complete df")
        print(df.SimpleModelLabel.value_counts(normalize=True))
        print("Scanner in train df")
        print(
            df[df.empi_anon.isin(train_id)].SimpleModelLabel.value_counts(
                normalize=True
            )
        )
        print(df[df.empi_anon.isin(train_id)].tissueden.value_counts(normalize=True))
        print("Scanner in test df")
        print(
            df[df.empi_anon.isin(test_id)].SimpleModelLabel.value_counts(normalize=True)
        )
        print(df[df.empi_anon.isin(test_id)].tissueden.value_counts(normalize=True))
        return {
            "train": df.loc[df.empi_anon.isin(train_id)],
            "val": df.loc[df.empi_anon.isin(val_id)],
            "test": df.loc[df.empi_anon.isin(test_id)],
        }

    def create_datasets(self) -> None:
        full_df = get_embed_csv()
        # Keep senograph essential a hold-out set
        self.full_df = full_df[full_df["SimpleModelLabel"] != 5]

        split_csv_dict = self.get_all_csv_from_config(
            orig_df=self.full_df,
            label=self.config.data.label,
            exclude_cviews=self.config.data.exclude_cviews,
            domain=self.config.data.domain,
            model=None,
        )

        self.target_size = self.config.data.augmentations.resize
        self.dataset_train = EmbedDataset(
            df=split_csv_dict["train"],
            transform=self.train_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=self.config.data.cache,
            use_counterfactuals=self.config.data.use_counterfactuals,
            counterfactual_contrastive_pairs=self.config.data.counterfactual_contrastive,
        )

        self.dataset_val = EmbedDataset(
            df=split_csv_dict["val"],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=self.config.data.cache,
        )

        self.dataset_test = EmbedDataset(
            df=split_csv_dict["test"],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=False,
        )

    @property
    def num_classes(self) -> int:
        match self.config.data.label:
            case "tissueden":
                return 4
            case "SimpleModelLabel":
                return 5
            case "ViewLabel":
                return 2
            case "CviewLabel":
                return 2
            case _:
                raise ValueError


class EmbedOODDataModule(EmbedDataModule):
    """
    Module to load the senograph essential domain from EMBED.
    We keep this as hold-out set distinct from pretraining domain
    for additional OOD evaluation.
    """

    def create_datasets(self):
        self.target_size = self.config.data.augmentations.resize
        full_df = get_embed_csv()
        df = full_df[full_df["SimpleModelLabel"] == 5]
        df = df.loc[df.FinalImageType == "2D"]

        y = df.groupby("empi_anon")["tissueden"].unique().apply(lambda x: x[0]).values

        train_val_id, test_id = train_test_split(
            df.empi_anon.unique(), test_size=0.25, random_state=33, stratify=y
        )

        train_id, val_id = train_test_split(
            df.empi_anon.unique(), test_size=0.10, random_state=33
        )

        print(
            f"N patients train: {train_id.shape[0]}, val: {val_id.shape[0]}, test {test_id.shape[0]}"
        )  # noqa

        if self.config.data.prop_train < 1.0:
            train_id, _ = train_test_split(
                train_id,
                train_size=int(self.config.data.prop_train * train_id.shape[0]),
                random_state=self.config.seed,
            )

        print("train n images" + str(len(df.loc[df.empi_anon.isin(train_id)])))
        print("test" + str(len(df.loc[df.empi_anon.isin(test_id)])))
        print("val" + str(len(df.loc[df.empi_anon.isin(val_id)])))
        print(df[df.empi_anon.isin(train_id)].tissueden.value_counts(normalize=True))
        print("Scanner in test df")
        print(df[df.empi_anon.isin(test_id)].tissueden.value_counts(normalize=True))

        split_csv_dict = {
            "train": df.loc[df.empi_anon.isin(train_id)],
            "val": df.loc[df.empi_anon.isin(val_id)],
            "test": df.loc[df.empi_anon.isin(test_id)],
        }

        self.dataset_train = EmbedDataset(
            df=split_csv_dict["train"],
            transform=self.train_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=self.config.data.cache,
            use_counterfactuals=self.config.data.use_counterfactuals,
            counterfactual_contrastive_pairs=self.config.data.counterfactual_contrastive,
        )

        self.dataset_val = EmbedDataset(
            df=split_csv_dict["val"],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=self.config.data.cache,
        )

        self.dataset_test = EmbedDataset(
            df=split_csv_dict["test"],
            transform=self.val_tsfm,
            target_size=self.target_size,
            label=self.config.data.label,
            parents=self.parents,
            cache=True,
        )


def prepare_vindr_dataset():
    df = pd.read_csv(VINDR_MAMMO_DIR / "breast-level_annotations.csv")
    meta = pd.read_csv(VINDR_MAMMO_DIR / "metadata.csv")
    df["img_path"] = df[["study_id", "image_id"]].apply(
        lambda x: VINDR_MAMMO_DIR / "pngs" / x[0] / f"{x[1]}.png", axis=1
    )
    df["ViewLabel"] = df.view_position.apply(lambda x: 0 if x == "MLO" else 1)
    tissue_maps = {"DENSITY A": 0, "DENSITY B": 1, "DENSITY C": 2, "DENSITY D": 3}
    df["tissueden"] = df.breast_density.apply(lambda x: tissue_maps[x])
    df = pd.merge(df, meta, left_on="image_id", right_on="SOP Instance UID")
    df["Scanner"] = df["Manufacturer's Model Name"]
    df["Manufacturer"] = df["Manufacturer"]
    df.dropna(subset="tissueden", inplace=True)
    return df


class VinDrDataModule(BaseDataModuleClass):
    def create_datasets(self):
        df = prepare_vindr_dataset()
        study_ids = df["Series Instance UID"].unique()
        train_val_id, test_id = train_test_split(
            study_ids, test_size=0.3, random_state=33
        )
        train_id, val_id = train_test_split(
            train_val_id, test_size=0.2, random_state=33
        )
        self.target_size = self.config.data.augmentations.resize
        print(
            f"N patients train: {train_id.shape[0]}, val: {val_id.shape[0]}, test {test_id.shape[0]}"
        )  # noqa
        if self.config.data.prop_train < 1.0:
            train_id = np.sort(train_id)
            y = (
                df.loc[df["Series Instance UID"].isin(train_id)]
                .groupby("Series Instance UID")["tissueden"]
                .unique()
                .apply(lambda x: x[0])
                .sort_index()
            )
            assert y.index[0] == train_id[0]
            train_id, _ = train_test_split(
                train_id,
                train_size=int(self.config.data.prop_train * train_id.shape[0]),
                stratify=y.values,
                random_state=self.config.seed,
            )

        self.dataset_train = VinDRMammoDataset(
            df=df.loc[df["Series Instance UID"].isin(train_id)],
            transform=self.train_tsfm,
            cache=self.config.data.cache,
            target_size=self.target_size,
        )

        self.dataset_val = VinDRMammoDataset(
            df=df.loc[df["Series Instance UID"].isin(val_id)],
            transform=self.val_tsfm,
            cache=self.config.data.cache,
            target_size=self.target_size,
        )

        self.dataset_test = VinDRMammoDataset(
            df=df.loc[df["Series Instance UID"].isin(test_id)],
            transform=self.val_tsfm,
            cache=True,
            target_size=self.target_size,
        )

    @property
    def dataset_name(self):
        return "vindr"

    @property
    def num_classes(self):
        return 4


class VinDRMammoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        target_size,
        cache: bool = True,
    ) -> None:
        self.imgs_paths = df.img_path.values
        print(f"len df {self.imgs_paths.shape[0]}")
        self.labels = df["tissueden"].values
        print(df["tissueden"].value_counts())
        print(df["tissueden"].value_counts(normalize=True))
        self.transform = transform
        self.target_size = target_size
        self.views = df.ViewLabel.values
        self.scanner = df.Scanner.values
        self.manufacturer = df.Manufacturer.values
        self.densities = df.tissueden.values
        data_dims = [1, self.target_size[0], self.target_size[1]]
        if cache:
            self.cache = SharedCache(
                size_limit_gib=96,
                dataset_len=self.labels.shape[0],
                data_dims=data_dims,
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __getitem__(self, index) -> Any:
        if self.cache is not None:
            # retrieve data from cache if it's there
            img = self.cache.get_slot(index)
            # x will be None if the cache slot was empty or OOB
            if img is None:
                img = preprocess_breast(self.imgs_paths[index], self.target_size)
                self.cache.set_slot(index, img, allow_overwrite=True)  # try to cache x
        else:
            img = preprocess_breast(self.imgs_paths[index], self.target_size)

        sample = {}
        sample["view"] = self.views[index]
        sample["y"] = self.labels[index]
        sample["scanner"] = self.scanner[index]
        sample["manufacturer"] = self.manufacturer[index]
        img = self.transform(img).float()
        sample["x"] = img
        return sample

    def __len__(self):
        return self.labels.shape[0]
