import joblib
import pydicom
import numpy as np
import os
import PIL
from PIL import Image
import math
from tqdm import tqdm
import logging as l
from glob import glob

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "window_center": img_dicom.WindowCenter,
        "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder+name+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def prepare_image(img_path):
    img_dicom = pydicom.dcmread(img_path)
    img_id = get_id(img_path)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img_id, img

def prepare_and_save(img_path, subfolder):
    try:
        img_id, img_pil = prepare_image(img_path)
        save_img(img_pil, subfolder, img_id)
    except KeyboardInterrupt:
        # Rais interrupt exception so we can stop the cell execution
        # without shutting down the kernel.
        raise
    except Exception as e:
        l.error(f"Error processing the image: {img_path}, Error: {e}")

def prepare_images(imgs_path, subfolder):
    for i in tqdm.tqdm(imgs_path):
        prepare_and_save(i, subfolder)

def prepare_images_njobs(img_paths, subfolder, n_jobs=-1):
    #print(subfolder)
    #joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(i, subfolder) for i in tqdm(img_paths))
    tasks = [joblib.delayed(prepare_and_save)(i, subfolder) for i in tqdm(img_paths)]
    joblib.Parallel(n_jobs=n_jobs)(tasks)
    #for path in tqdm(img_paths):
    #    prepare_and_save(path, subfolder)

if __name__ == '__main__':

    dcm_path = '/app/dcm'
    png_path = '/app/temppng'

    if not os.path.exists(png_path):
        os.makedirs(png_path)

    prepare_images_njobs(glob(dcm_path+'/*'), png_path+'/')
