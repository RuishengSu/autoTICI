import argparse

import SimpleITK as sitk
import ants
import cv2 as cv
import numpy as np

import registration.transformation
from utils.utils import normalize, extract_skull_mask


def register(img_fixed, img_moving):
    """Main function"""
    '''Input'''
    if isinstance(img_fixed, str):
        img_fixed = np.load(img_fixed)
    if isinstance(img_moving, str):
        img_moving = np.load(img_moving)

    '''Blur images'''
    img_fixed = cv.GaussianBlur(img_fixed, (25, 25), 0)
    img_moving = cv.GaussianBlur(img_moving, (25, 25), 0)

    '''Settings'''
    transform = 'affine'
    resolution = 6
    n_iteration = ['512']
    elastix_metric = ['AdvancedMattesMutualInformation']
    ants_metric = 'MattesMutualInformation'
    final_resample_interpolator = ['FinalLinearInterpolator']
    interpolator = ['LinearInterpolator']
    final_BSpline_interpolation_order = ['1']

    parameterMap = sitk.GetDefaultParameterMap(transform, resolution)
    # parameterMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
    parameterMap['Metric'] = elastix_metric
    parameterMap['Interpolator'] = interpolator
    parameterMap['MaximumNumberOfIterations'] = n_iteration
    parameterMap['ResampleInterpolator'] = final_resample_interpolator
    parameterMap['FinalBSplineInterpolationOrder'] = final_BSpline_interpolation_order
    parameterMap['ErodeMask'] = ['false']
    parameterMap['RequiredRatioOfValidSamples'] = ['0.05']

    parameterMap['NumberOfResolutions'] = ['4']
    parameterMap['ImagePyramidSchedule'] = ['8', '8', '4', '4', '2', '2', '1', '1']

    '''Execution'''
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(img_fixed))
    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(img_moving))

    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()

    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    result_img = np.array(sitk.GetArrayFromImage(elastixImageFilter.GetResultImage()), dtype=np.uint8)

    '''Get mattes mi value'''
    met = ants.create_ants_metric(ants.from_numpy(img_fixed), ants.from_numpy(result_img), metric_type=ants_metric)
    metric_value = abs(met.get_value())

    return metric_value, transformParameterMap


def register_to_postEVT(preEVT_sequence, postEVT_sequence):
    """Register preEVT to postEVT"""
    preEVT_sequence, postEVT_sequence = normalize(preEVT_sequence), normalize(postEVT_sequence)
    '''Extract skull mask'''
    skull_masked_minip_preEVT, _, _ = extract_skull_mask(preEVT_sequence)
    skull_masked_minip_postEVT, _, _ = extract_skull_mask(postEVT_sequence)
    _, transform_parameter_for_preEVT = register(skull_masked_minip_postEVT, skull_masked_minip_preEVT)
    preEVT_sequence = registration.transformation.transform_sequence(preEVT_sequence, transform_parameter_for_preEVT)

    return preEVT_sequence, postEVT_sequence

def parse_args():
    """
    Argument parser for the main function
    """
    parser = argparse.ArgumentParser(description='Register a pair of images')
    parser.add_argument('fixed', type=str, help='Input fixed image file path')
    parser.add_argument('moving', type=str, help='Input moving image file path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    register(args.fixed, args.moving)
