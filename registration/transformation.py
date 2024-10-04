import SimpleITK as sitk
import cv2 as cv


def transform_sequence(img_seq, transform_parameter_map):
    transformed_img_seq = img_seq.copy()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    for i, img in enumerate(img_seq[:]):
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
        transformed_img_seq[i] = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img_seq


def transform_image(img, transform_parameter_map):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    transformixImageFilter.SetTransformParameterMap(transform_parameter_map)
    transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(img))
    transformed_img = sitk.GetArrayFromImage(transformixImageFilter.Execute())
    return transformed_img


def warp_sequence(img, transformation_matrix):
    frames, rows, cols = img.shape
    for frame_idx in range(frames):
        img[frame_idx, :, :] = cv.warpAffine(img[frame_idx, :, :], transformation_matrix, (cols, rows))
    return img
