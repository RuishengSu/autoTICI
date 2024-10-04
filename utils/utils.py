import logging
from pathlib import Path
import cv2 as cv
import numpy as np
import pydicom
from scipy.ndimage import binary_opening, uniform_filter
from scipy.stats import mode
from skimage.morphology import binary_dilation, disk, remove_small_objects, binary_closing, remove_small_holes
from skimage.transform import resize
from skimage.filters import frangi
import config
logger = logging.getLogger(__name__)


def normalize(img):
    """Outputs image of type unsigned int"""
    image_minip_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    return image_minip_norm.astype(np.uint8)


def normalize_0_1(img):
    """Outputs image of type unsigned int"""
    image_minip_norm = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return image_minip_norm


def remove_text_and_border(in_img):
    def remove_text_2d(img, text_inpaint_radius=3, border_inpaint_radius=10, border_margin=3):
        """
        Input: expected image range 0-255; if 2D+t, t is expected to be the first channel.
        Remove text and black borderlines from a 2D image
        border_margin: borderline dilation radius"""
        '''text mask'''
        black_text_mask = np.zeros([*img.shape], dtype=bool)
        black_text_mask[img < config.MIN_VESSEL_INTENSITY] = True
        black_text_mask = binary_dilation(black_text_mask, disk(10))

        white_text_mask = np.zeros([*img.shape], dtype=bool)
        white_text_mask[img > config.MAX_VALID_INTENSITY] = True
        white_text_mask = binary_dilation(white_text_mask, disk(10))

        text_mask = white_text_mask & black_text_mask
        text_mask = binary_dilation(text_mask, disk(5))

        '''black border line mask'''
        black_border_mask = np.zeros(img.shape, dtype=bool)
        black_border_mask[img < config.MIN_VESSEL_INTENSITY] = True
        black_border_mask = remove_small_objects(black_border_mask, min_size=500, connectivity=2)
        black_border_mask = binary_dilation(black_border_mask, disk(border_margin))

        '''Combined mask'''
        mask = (text_mask | black_border_mask)
        '''inpaint with small radius for less blur in text area'''
        img = cv.inpaint(img, mask.astype(np.uint8), text_inpaint_radius, flags=cv.INPAINT_TELEA)
        '''inpaint with larger radius for less noise in border area'''
        img = cv.inpaint(img, black_border_mask.astype(np.uint8), border_inpaint_radius, flags=cv.INPAINT_TELEA)
        return img

    out_img = in_img.copy()
    dim = len(in_img.shape)
    if dim == 3:
        for idx, frame in enumerate(in_img):
            out_img[idx] = remove_text_2d(frame)
    elif dim == 2:
        out_img = remove_text_2d(in_img)
    return out_img


def minip(img_seq, axis=0):
    return np.min(img_seq, axis=axis)


def read_sequence(fp):
    sequence = pydicom.dcmread(fp, defer_size="1 KB", stop_before_pixels=False, force=True)
    pixel_spacing = get_pixel_spacing_from_header(sequence)
    sequence = sequence.pixel_array
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)
    return sequence, pixel_spacing


def get_pixel_spacing_from_header(ds):
    if 'PixelSpacing' in ds:
        return ds.PixelSpacing[0]
    if ('DistanceSourceToDetector' in ds) and ('DistanceSourceToPatient' in ds) and ('ImagerPixelSpacing' in ds):
        imager_pixel_spacing = np.array([float(s) for s in ds.ImagerPixelSpacing], dtype='float32')
        pixel_spacing = imager_pixel_spacing * ds.DistanceSourceToPatient / ds.DistanceSourceToDetector
        return pixel_spacing[0]
    return np.nan


def resize_to_1024(seq, pixel_spacing):
    if seq.shape[1:] != (1024, 1024):
        logger.info("Resizing frames from {} to 1024*1024, "
                    "pixel spacing from {} to {}".format(seq.shape[1:], pixel_spacing,
                                                         pixel_spacing * (seq.shape[1] / 1024)))
        seq = pad_sequence(seq, to=max(seq.shape[1:]))  # pad frames to square images
        if seq.shape[1] != 1024:
            pixel_spacing *= (seq.shape[1] / 1024)
            seq = resize(seq, (seq.shape[0], 1024, 1024), anti_aliasing=False, preserve_range=True)
        seq = np.array(seq, dtype=np.uint8)
    return seq, pixel_spacing


def extract_skull_mask(sequence):
    def diffcount(A):
        B = A.copy()
        B.sort(axis=0)
        C = np.diff(B, axis=0) != 0
        D = C.sum(axis=0) + 1
        return D

    sequence = normalize(sequence.copy())
    img = minip(sequence)

    unique_value_img = diffcount(sequence)
    background_mask = np.zeros_like(unique_value_img, dtype=bool)
    if sequence.shape[0] >= 2:
        background_mask = (unique_value_img == 1) & (img != 0)

    background_mask = binary_opening(background_mask)
    img[background_mask] = 0

    '''Try to get background intensity value. If not succeed, use 255 as default.'''
    background_intensity = 255  # assuming img intensity range [0, 255]
    if np.count_nonzero(background_mask) != 0:
        background_intensity = np.median(np.min(sequence[:, background_mask], axis=0))
    return img, background_intensity, background_mask


def segment_minip(preEVT_sequence, postEVT_sequence, TDT_mask):
    preEVT_sequence, postEVT_sequence = normalize(preEVT_sequence), normalize(postEVT_sequence)
    minip_preEVT, minip_postEVT = minip(preEVT_sequence), minip(postEVT_sequence)

    _, background_preEVT, background_mask_preEVT = extract_skull_mask(preEVT_sequence)
    _, background_postEVT, background_mask_postEVT = extract_skull_mask(postEVT_sequence)

    vessel_mask_preEVT, perfusion_map_preEVT, hist_preEVT = perfusion_segmentation(minip_preEVT, background=background_preEVT)
    vessel_mask_postEVT, perfusion_map_postEVT, hist_postEVT = perfusion_segmentation(minip_postEVT, background=background_postEVT)

    preEVT_TDT = (~perfusion_map_preEVT) & (TDT_mask > 0) & (minip_preEVT > config.MIN_VESSEL_INTENSITY)
    postEVT_reperfusion_mask = preEVT_TDT & perfusion_map_postEVT
    count_TDT_mask = np.count_nonzero(TDT_mask)
    count_preEVT_TDT = np.count_nonzero(preEVT_TDT)
    count_reperfusion = np.count_nonzero(postEVT_reperfusion_mask)
    count_postEVT_TDT = np.count_nonzero(preEVT_TDT) - count_reperfusion
    if count_preEVT_TDT <= 0.05 * count_TDT_mask:
        logger.warning("Way too small preEVT TDT")
        autoTICI = 1
    else:
        autoTICI = count_reperfusion / count_preEVT_TDT

    '''calculate autoTICI based on only postEVT image'''
    post_autoTICI_TDT = (~perfusion_map_postEVT) & (TDT_mask > 0) & (minip_postEVT > config.MIN_VESSEL_INTENSITY)

    '''Generate vis for pipeline'''
    minip_preEVT = np.dstack([minip_preEVT, minip_preEVT, minip_preEVT])
    minip_postEVT = np.dstack([minip_postEVT, minip_postEVT, minip_postEVT])

    TDT_mask_contour_points, _ = cv.findContours(TDT_mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(minip_preEVT, TDT_mask_contour_points, -1, (0, 0, 255), thickness=12)
    preEVT_TDT_contour_points, _ = cv.findContours(preEVT_TDT.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(minip_preEVT, preEVT_TDT_contour_points, -1, (255, 255, 255), thickness=8)

    cv.drawContours(minip_postEVT, preEVT_TDT_contour_points, -1, (255, 255, 255), thickness=-1)
    cv.drawContours(minip_postEVT, preEVT_TDT_contour_points, -1, (255, 255, 255), thickness=8)

    postEVT_reperfusion_contours, _ = cv.findContours(postEVT_reperfusion_mask.astype(np.uint8), cv.RETR_TREE,
                                                      cv.CHAIN_APPROX_NONE)
    cv.drawContours(minip_postEVT, postEVT_reperfusion_contours, -1, (0, 140, 255), thickness=-1)

    label_image = cv.putText(img=np.zeros((100, 1024, 3)), text="Scoring", org=(350, 70),
                             fontFace=3, fontScale=3, color=(255, 255, 255), thickness=5)
    vis = np.concatenate([label_image, minip_preEVT, minip_postEVT])

    return autoTICI, vis


def binarize_image(img, thresh=0):
    # Otsu's thresholding after Gaussian filtering. Inpput image must by of dtype: uint8
    if thresh != 0:
        return cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
    else:
        return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


def save_fig(fig_save_path, *img):
    Path(fig_save_path).parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(fig_save_path, np.concatenate([*img]))


def truncate(img, img_min=0, img_max=255):
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img


def perfusion_segmentation(img, background=255):
    img_vessel = frangi(img.astype(float), sigmas=(2, 12, 2))
    img_vessel = img_vessel * 255
    img_vessel = truncate(img_vessel, img_min=0, img_max=255)
    _, img_vessel_binary = binarize_image(img_vessel.astype(np.uint8), thresh=config.FRONGI_INTENSITY_THRES)
    # a boolean array of (width, height) in which False is invalid pixels (vessel) and True is valid pixels (non-vessel)
    vessel_mask = (img_vessel_binary == 255) & (img >= config.MIN_VESSEL_INTENSITY)

    '''Get all non-vessel pixel values'''
    '''Temporary workaround for the transformation artifacts on the image edge area'''
    data = np.ravel(img[(~vessel_mask) & (img >= config.MIN_PERFUSION_INTENSITY) & (img <= background)]).astype(
        np.uint8)
    ret2, _ = binarize_image(data, thresh=0)  # otsu

    perfused = (img >= config.MIN_PERFUSION_INTENSITY) & (img <= ret2) & (~vessel_mask)

    '''Calculate perfusion map'''
    '''1. divide vessel pixels to either perfused or non-perfused'''
    perfusion_map = np.zeros(img.shape)
    perfusion_map[perfused] = 1
    perfusion_map[vessel_mask] = 0.5
    perfusion_map = uniform_filter(perfusion_map, size=50)
    perfusion_map = (perfusion_map >= 0.5)
    '''2. remove small holes or small disconnected areas'''
    perfusion_map = binary_closing(perfusion_map, disk(10))
    perfusion_map = remove_small_holes(perfusion_map, area_threshold=1000)
    perfusion_map = remove_small_objects(perfusion_map, min_size=1000)

    hist_data = data
    return vessel_mask, perfusion_map, hist_data


def pad_image(img, to=1024, cval=None):
    pad_h1 = (to - img.shape[0]) // 2
    pad_h2 = to - img.shape[0] - pad_h1
    pad_w1 = (to - img.shape[1]) // 2
    pad_w2 = to - img.shape[1] - pad_w1
    if cval is None:
        cval, _ = mode(img, axis=None, keepdims=False)
    return np.pad(img, ((pad_h1, pad_h2), (pad_w1, pad_w2)), 'constant', constant_values=cval)


def pad_sequence(seq, to=1024):
    out = []
    for i in range(seq.shape[0]):
        out.append(pad_image(seq[i], to=to))
    return np.stack(out, axis=0)


def resize_to_target_spacing(seq, pixel_spacing, target_spacing=None, masks=None):
    if target_spacing is None:
        target_spacing = (0.15, abs(pixel_spacing))[0.14 <= abs(pixel_spacing) <= 0.16]
    if target_spacing != abs(pixel_spacing):
        seq_new_size = int(1024 * abs(pixel_spacing) / target_spacing)
        seq = resize(seq, (seq.shape[0], seq_new_size, seq_new_size), anti_aliasing=False, preserve_range=True)
        seq = np.array(seq, dtype=np.uint8)
        if masks is not None:
            for i, mask in enumerate(masks):
                masks[i] = resize(mask, (seq_new_size, seq_new_size), anti_aliasing=False, preserve_range=True)
        if seq.shape[1] < 1024:
            seq = pad_sequence(seq, to=1024)
            if masks is not None:
                for i, mask in enumerate(masks):
                    masks[i] = pad_image(mask, to=1024, cval=0)

        if seq.shape[1] > 1024:
            crop_size = (seq.shape[1] - 1024) // 2
            seq = seq[:, crop_size:crop_size + 1024, crop_size:crop_size + 1024]
            if masks is not None:
                for i, mask in enumerate(masks):
                    masks[i] = mask[crop_size:crop_size + 1024, crop_size:crop_size + 1024]
        pixel_spacing = target_spacing
    return seq, pixel_spacing
