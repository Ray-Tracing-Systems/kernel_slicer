import os

import utils
from logger import Log, Status

mse_threshold  = 1e-4
mse_threshold2 = 1e-2


def is_out_img(img_name: str):
    return img_name.startswith("zout_") and utils.has_image_ext(img_name)


def find_image_pairs():
    filenames = utils.get_files(os.getcwd())
    image_filenames = sorted([f for f in filenames if is_out_img(f)])
    cpu_images = [img for img in image_filenames if img.find("cpu") >= 0]
    gpu_images = [img for img in image_filenames if img.find("gpu") >= 0]
    if len(cpu_images) != len(gpu_images):
        Log().error("Non equal image count for different code versions: cpu={0}, gpu={1}".format(
            len(cpu_images), len(gpu_images)
        ))
        return None

    return list(zip(cpu_images, gpu_images))


def compare_images(img_name1, img_name2):
    mse_res = utils.load_and_calc_mse(img_name1, img_name2)
    status = Status.OK 
    if mse_res > mse_threshold2:
      status = Status.FAILED
    elif mse_res > mse_threshold:
      status = Status.WARNING
    Log().status_info("{0}, {1} | mse = {2}".format(img_name1, img_name2, mse_res), status=status)
    return status


def compare_generated_images():
    Log().info("Comparing images")
    image_pairs = find_image_pairs()
    if image_pairs is None:
        return -1

    status = Status.OK
    for img1, img2 in image_pairs:
        status = Status.worst_of([status, compare_images(img1, img2)])

    return status

