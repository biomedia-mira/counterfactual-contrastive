import pandas as pd
from pathlib import Path
from skimage import io
from torchvision.transforms import Resize, ToTensor, ToPILImage
from tqdm import tqdm
from multiprocessing import Pool

if __name__ == "__main__":
    PADCHEST_ROOT = Path("/vol/biodata/data/chest_xray/BIMCV-PADCHEST")
    PADCHEST_IMAGES = PADCHEST_ROOT / "images"
    TARGET = Path("/data/PadChest/preprocessed")
    df = pd.read_csv(
        PADCHEST_ROOT / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    )
    df = df.loc[df.Projection == "PA"]
    df = df.loc[df.Pediatric == "No"]
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
    img_paths = df.ImageID.values
    resize = Resize(512, antialias=True)
    to_tensor = ToTensor()
    to_pil = ToPILImage(mode="L")

    def resize_image(path):
        img = io.imread(PADCHEST_IMAGES / path, as_gray=True)
        img = img / (img.max() + 1e-12)
        img_out = resize(to_tensor(img))
        img_out = to_pil(img_out)
        img_out.save(TARGET / path)

    pool = Pool(processes=112)
    for _ in tqdm(pool.imap_unordered(resize_image, img_paths), total=len(img_paths)):
        pass
