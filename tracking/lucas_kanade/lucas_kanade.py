import numpy as np
import cv2

from tqdm import tqdm

def get_roi_points(roi):
    roi_top_left = roi[0]
    roi_top_right = [roi[1][0], roi[0][1]]
    roi_bottom_left = [roi[0][0], roi[1][1]]
    roi_bottom_right = roi[1]

    return np.array([roi_top_left, roi_top_right, roi_bottom_left, roi_bottom_right])

def lucas_kanade(frames, roi, max_iter=1):
    # get initial roi points
    roi = get_roi_points(roi)
    rois = [roi]

    # start loop through frames
    previous_frame = frames[0]
    for frame in tqdm(frames[1:]):

        # initizize params
        params = np.zeros(6)

        num_iter = 0
        
        while True:

            # get rectangular bounding box to cut template
            roi = ((roi[:, 0].min(), roi[:, 1].min()), (roi[:, 0].max(), roi[:, 1].max()))
            roi = get_roi_points(roi)

            # get template
            template = previous_frame[roi[0][1]:roi[2][1], roi[0][0]:roi[1][0]] 

            # warp new frame
            height, width = frame.shape
            params_ = np.array([[1.0 + params[0], 0.0 + params[2], 0.0 + params[4]],
                            [0.0 + params[1], 1.0 + params[3], 0.0 + params[5]]])
            warped_frame = cv2.warpAffine(frame, params_, (width, height), cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_template = warped_frame[roi[0][1]:roi[2][1], roi[0][0]:roi[1][0]]

            # compute error
            error = cv2.subtract(template, warped_template)

            # compute gradient on new frame
            gradient_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)

            # get roi
            gradient_x = gradient_x[roi[0][1]:roi[2][1], roi[0][0]:roi[1][0]]
            gradient_y = gradient_y[roi[0][1]:roi[2][1], roi[0][0]:roi[1][0]]

            # warp gradients
            height, width = gradient_x.shape
            warped_gradient_x = cv2.warpAffine(gradient_x, params_, (width, height), cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            height, width = gradient_y.shape
            warped_gradient_y = cv2.warpAffine(gradient_y, params_, (width, height), cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # stack gradients
            warped_gradient = np.dstack([warped_gradient_x, warped_gradient_y])

            # compute jaccobian
            jacobbian = np.zeros((template.shape[0], template.shape[1], 2, 6))
            for y in range(template.shape[0]):
                for x in range(template.shape[1]):
                    jacobbian[y, x] = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])

            # compute steepest
            steepest = np.matmul(np.expand_dims(warped_gradient, -2), jacobbian)

            # compute hessian
            hessian = np.matmul(np.swapaxes(steepest, 2, 3), steepest).sum(axis=(0, 1))

            # compute invers hessian 
            inv_hessian = np.linalg.inv(hessian)

            # compute delta for parameters
            params_delta = np.matmul(inv_hessian, (np.swapaxes(steepest, 2, 3)[..., 0] * np.expand_dims(error, -1)).sum(axis=(0, 1)))

            # update parameters
            params += params_delta

            # check condition
            if num_iter > max_iter:
                break

            previous_frame = frame.copy()
            num_iter += 1
        
        # warp roi points
        params_ = np.array([[1.0 + params[0], 0.0 + params[2], 0.0 + params[4]],
                            [0.0 + params[1], 1.0 + params[3], 0.0 + params[5]]])
        roi = np.apply_along_axis(lambda x: np.matmul(params_, np.hstack((x, 1.0))), 1, rois[-1])
        roi = roi.astype(int)   
        roi[roi < 0] = 0
        rois.append(roi)

    return rois


        




        

