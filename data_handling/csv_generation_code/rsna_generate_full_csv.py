#  Most function in this file are adapted from the Active Label Cleaning paper code:
# https://github.com/microsoft/InnerEye-DeepLearning/blob/
# /InnerEye-DataQuality/InnerEyeDataQuality/datasets/noisy_cxr_benchmark_creation/
# create_noisy_chestxray_dataset.py

import json
from pathlib import Path

import pandas as pd


def create_nih_dataframe(
    mapping_file_path: Path, nih_metadata_file: Path
) -> pd.DataFrame:
    """
    This function loads the json file mapping NIH ids to Kaggle images.
    Loads the original NIH label (multiple labels for each image).
    :param mapping_file_path: path to the json mapping from NIH to Kaggle dataset (on the
    RSNA webpage)
    :return: dataframe with original NIH labels for each patient in the Kaggle dataset.
    """
    with open(mapping_file_path) as f:
        list_subjects = json.load(f)
    orig_dataset = pd.DataFrame(columns=["patientId", "orig_label"])
    orig_dataset["patientId"] = [subj["subset_img_id"] for subj in list_subjects]
    orig_dataset["nih_image_id"] = [subj["img_id"] for subj in list_subjects]
    orig_labels = [str(subj["orig_labels"]).lower() for subj in list_subjects]
    orig_dataset["orig_label"] = orig_labels
    orig_dataset["orig_label"].apply(lambda x: sorted(x))
    orig_dataset["StudyInstanceUID"] = [
        subj["StudyInstanceUID"] for subj in list_subjects
    ]
    metadata_nih = pd.read_csv(nih_metadata_file)
    metadata_nih.rename(columns={"Image Index": "nih_image_id"}, inplace=True)
    metadata_nih = metadata_nih[
        [
            "nih_image_id",
            "Follow-up #",
            "Patient Age",
            "Patient Gender",
            "View Position",
        ]
    ]
    return pd.merge(orig_dataset, metadata_nih)


def create_mapping_dataset_nih(
    mapping_file_path: Path,
    nih_metadata: Path,
    kaggle_dataset_path: Path,
) -> pd.DataFrame:
    """
    Creates the final chest x-ray dataset combining labels from NIH and kaggle
    Args:
        mapping_file_path: json to map RSNA ids to NIH original Ids
        nih_metadata: file to NIH metadata csv
        kaggle_dataset_path: train labels csv RSNA Pneumonia Challenge

    Returns:
        dataset with all metadata, Kaggle label
    """
    orig_dataset = create_nih_dataframe(mapping_file_path, nih_metadata)
    kaggle_dataset = pd.read_csv(kaggle_dataset_path)
    # Merge NIH info with Kaggle dataset
    merged = pd.merge(orig_dataset, kaggle_dataset)
    merged.rename(columns={"Target": "label_rsna_pneumonia"}, inplace=True)
    merged.label_rsna_pneumonia = merged.label_rsna_pneumonia.astype(bool)
    merged.drop(columns=["x", "y", "width", "height"], inplace=True)
    merged.drop_duplicates(subset="patientId", inplace=True)
    return merged


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    mapping_file = (
        Path(__file__).parent
        / "original_csvs"
        / "pneumonia-challenge-dataset-mappings_2018.json"
    )
    nih_metadata = Path(__file__).parent / "original_csvs" / "Data_Entry_2017.csv"
    kaggle_dataset_path = current_dir / "original_csvs" / "stage_2_train_labels.csv"
    dataset = create_mapping_dataset_nih(
        mapping_file,
        nih_metadata,
        kaggle_dataset_path,
    )
    dataset.to_csv(current_dir / "pneumonia_dataset_with_metadata.csv", index=False)
