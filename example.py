import cv2
import toml
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from edge_aware_surface_normal_calculator import CameraParameter, DepthFrame, SurfaceNormalCalculator

WORK_DIR = str(Path().resolve())


def read_toml(toml_path: str) -> Dict[str, Any]:
    with open(toml_path, "r") as f:
        dict_toml = toml.load(f)
    return dict_toml


def truncate_far_depth(image: np.ndarray, depth_threshold=2000) -> np.ndarray:
    assert len(image.shape) == 2
    image[image > depth_threshold] = 0
    return image


def colorize_depth_image(image, max_var=2000):
    assert len(image.shape) == 2
    assert image.dtype == np.int16 or image.dtype == np.uint16

    image_colorized = np.zeros([image.shape[0], image.shape[1], 3]).astype(np.uint8)
    image_colorized[:, :, 1] = 255
    image_colorized[:, :, 2] = 255

    image_hue = image.copy().astype(np.float32)
    image_hue[np.where(image_hue > max_var)] = 0
    zero_idx = np.where((image_hue > max_var) | (image_hue == 0))
    image_hue *= 255.0 / max_var
    image_colorized[:, :, 0] = image_hue.astype(np.uint8)
    image_colorized = cv2.cvtColor(image_colorized, cv2.COLOR_HSV2RGB)
    image_colorized[zero_idx[0], zero_idx[1], :] = 0
    return image_colorized


def colorize_float_image(image, max_var=1.0):
    assert len(image.shape) == 3 and image.shape[2] == 3
    assert image.dtype == np.float32

    image_hue = image.copy()
    image_hue *= 255.0 / max_var
    image_hue = image_hue.astype(np.uint8)
    image_colorized = cv2.cvtColor(image_hue, cv2.COLOR_HSV2RGB)
    return image_colorized


def show_image(title: str, image: np.ndarray):
    while True:
        cv2.imshow(title, image)
        if cv2.waitKey(10) & 0xFF in [27, ord("q")]:
            break
    cv2.destroyAllWindows()


# Get Camera Parameters
toml_path = str(Path(WORK_DIR, "config", "config.toml"))
dict_toml = read_toml(toml_path)
camera_parameter = CameraParameter(
    fx=dict_toml["Camera"]["fx"],
    fy=dict_toml["Camera"]["fy"],
    cx=dict_toml["Camera"]["cx"],
    cy=dict_toml["Camera"]["cy"],
    image_width=dict_toml["Camera"]["width"],
    image_height=dict_toml["Camera"]["height"],
)

# Get Depth Image
depth_image_path = str(Path(WORK_DIR, "data", "depth.png"))
depth_image_raw = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
depth_image = truncate_far_depth(depth_image_raw)
depth_frame = DepthFrame(depth_image=depth_image, camera_parameter=camera_parameter)

# Get Surface Normal and Edge Images
surface_normal_calculator = SurfaceNormalCalculator(camera_parameter=camera_parameter)
start = time.time()
surface_normal_calculator.compute(depth_frame)
end = time.time()
print(f"elapsed:{end-start}")
edge_image = surface_normal_calculator.get_edge_image()
surface_normal_image = surface_normal_calculator.get_surface_normal_image()
surface_normal_image_colorized = colorize_float_image(surface_normal_image)

depth_image_colorized = colorize_depth_image(depth_image, max_var=2000)
concat_image_for_viz = cv2.hconcat([depth_image_colorized, cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR), surface_normal_image_colorized])
show_image("res", concat_image_for_viz)
