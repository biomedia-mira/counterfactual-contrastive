import numpy as np
import pydicom


def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    Taken from https://github.com/vinbigdata-medical/vindr-mammo/blob/master/visualize.py
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    if (
        ("WindowCenter" not in data)
        or ("WindowWidth" not in data)
        or ("PhotometricInterpretation" not in data)
        or ("RescaleSlope" not in data)
        or ("PresentationIntentType" not in data)
        or ("RescaleIntercept" not in data)
    ):
        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element("PresentationIntentType").value
    if str(intentType).split(" ")[-1] == "PROCESSING":
        print(f"{dicom_file} got processing file")
        return

    c = data.data_element("WindowCenter").value  # data[0x0028, 0x1050].value
    w = data.data_element("WindowWidth").value  # data[0x0028, 0x1051].value
    if isinstance(c, pydicom.multival.MultiValue):
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element("PhotometricInterpretation").value

    try:
        a = data.pixel_array
    except:  # noqa
        print(f"{dicom_file} Cannot get get pixel_array!")
        return

    slope = data.data_element("RescaleSlope").value
    intercept = data.data_element("RescaleIntercept").value
    a = a * slope + intercept

    try:
        pad_val = data.get("PixelPaddingValue")
        pad_limit = data.get("PixelPaddingRangeLimit", -99999)
        if pad_limit == -99999:
            mask_pad = a == pad_val
        else:
            if str(photometricInterpretation) == "MONOCHROME2":
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:  # noqa
        # Manually create padding mask
        # this is based on the assumption that padding values take
        # majority of the histogram
        print(f"{dicom_file} has no PixelPaddingValue")
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent
            # than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(
                    f"{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}"  # noqa
                )
        except:  # noqa
            print(f"{dicom_file} most frequent pixel value {sorted_pixels[0]}")

    # apply window
    mm = c - 0.5 - (w - 1) / 2
    MM = c - 0.5 + (w - 1) / 2
    a[a < mm] = 0
    a[a > MM] = 255
    mask = (a >= mm) & (a <= MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w - 1) + 0.5) * 255

    if str(photometricInterpretation) == "MONOCHROME1":
        a = 255 - a

    a[mask_pad] = 0
    return a


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    import cv2
    from tqdm import tqdm

    df = pd.read_csv("/vol/biodata/data/Mammo/VinDr-Mammo/breast-level_annotations.csv")
    study_ids = df.study_id.values
    image_ids = df.image_id.values
    orig_dir = Path("/vol/biodata/data/Mammo/VinDr-Mammo/images")
    target_dir = Path("/vol/biomedic3/data/VinDR-Mammo/pngs")

    for img_id, study_id in tqdm(zip(image_ids, study_ids)):
        rel_path = Path(study_id) / img_id
        orig_path = orig_dir / study_id / f"{img_id}.dicom"
        png_img = convert_dicom_to_png(orig_path)
        target_path = target_dir / study_id / f"{img_id}.png"
        target_path.parent.mkdir(exist_ok=True, parents=True)
        success = cv2.imwrite(str(target_path), png_img)
        try:
            assert success
        except AssertionError:
            print(target_path)
