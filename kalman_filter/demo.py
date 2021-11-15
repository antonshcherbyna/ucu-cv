import numpy as np
import cv2
import argparse
import time

from pynput.mouse import Controller
from kalman_filter import KalmanFilter


def demo(args):

    # create window
    frame = np.zeros((900, 1440, 3), np.uint8)
    
    # create mouse controller
    mouse = Controller()    

    # parameters for Kalman filter
    noise = args.noise
    noise_variance = args.noise_variance
    predict = args.predict

    # get initial state
    position_x, position_y = mouse.position
    initial_state = np.array([position_x, position_y, 0, 0])

    # create Kalman filter
    kalman_filter = KalmanFilter(initial_state=initial_state)

    # loop
    prev_estimated_position = None
    background = None
    while True:
        # measure
        new_position_x, new_position_y = mouse.position
        if noise:
            new_position_x = new_position_x + int(np.random.normal(0, args.noise_variance, 1))
            new_position_y = new_position_y + int(np.random.normal(0, args.noise_variance, 1))
        velocity_x = new_position_x - position_x 
        velocity_y = new_position_y - position_y 
        state = np.array([new_position_x, new_position_y, velocity_x, velocity_y])
        position_x = new_position_x
        position_y = new_position_y

        kalman_filter.predict()
        
        # calibrate & get estimated position
        estimated_state = kalman_filter.calibrate(state)
        estimated_position = estimated_state[:2]

        # get real position
        real_position = state[:2]

        # prepare data for opencv
        estimated_position = tuple(estimated_position.astype(int))
        real_position = tuple(real_position.astype(int))

        # draw real position
        cv2.circle(frame, real_position, 3, (255, 0, 0), -1)

        # draw smoothed estimated curve
        if prev_estimated_position is not None:
            cv2.line(frame, prev_estimated_position, estimated_position, (0, 255, 0), 3)
        prev_estimated_position = estimated_position
        
        # make prediction
        if predict:
            # make copy of a background
            background = frame.copy()

            # get prediction for velocity
            predicted_velocity = estimated_state[2:]
            predicted_position = estimated_state[:2] + 5*predicted_velocity

            # prepate data for opencv
            predicted_position = tuple(predicted_position.astype(int))

            # draw prediction for movement
            cv2.line(frame, estimated_position, predicted_position, (0, 0, 255), 3)

        # show 
        cv2.imshow("Mouse tracking", frame)
        k = cv2.waitKey(10) & 0XFF
        if background is not None:
            frame = background.copy()
            cv2.imshow("Mouse tracking", frame)

        cv2.destroyAllWindows()
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', action='store_true', default=False)
    parser.add_argument('--noise_variance', type=float, default=20.0)
    parser.add_argument('--predict', action='store_true', default=False)
    args = parser.parse_args()

    demo(args)