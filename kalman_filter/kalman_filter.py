import numpy as np

class KalmanFilter:
    
    def __init__(self, initial_state):
        
        # state transition matrix
        self.F = np.array([[1.0, 0.0, 0.2, 0.0],
                           [0.0, 1.0, 0.0, 0.2],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        
        # measurement matrix
        self.H = np.array([[1.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]])
        
        # action uncertainty matrix
        self.Q = np.array([[0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.1, 0.0],
                           [0.0, 0.0, 0.0, 0.1]])

        # sensor noise matrix
        self.R = 0.1 * np.eye(4)

        # covariance matrix
        self.P = np.zeros((4, 4))
        
        # state of the system
        self.x = initial_state

        
    def predict(self):
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        
        return self.x
        
    def calibrate(self, Z):
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R))
        self.x = self.x + np.matmul(K, (Z - np.matmul(self.H, self.x)))
        self.P = self.P - np.matmul(np.matmul(K, self.H), self.P)
        
        return self.x