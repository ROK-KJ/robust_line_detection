import cv2, numpy as np
import traceback
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2, yaml, os
from cost_functions import CostFunction
from utils import *

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('results.mp4', fourcc, 30.0, (640, 480))

POINT_DISTANCE_THRESHOLD=25 
HOUGH_LINE_THRESHOLD=25
MIN_LINE_LENGTH=50
MAX_LINE_GAP=40
lk_params = dict(winSize=(21, 21), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                           10, 0.03))

class LinePointTracker :
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.tracking_callback)
        self.old_gray = None
        self.frame_gray = None
        self.p0 = None
        self.best_params = None
        self.lines = None
        self.edges = None

    def tracking_callback(self, data, **best_params):
        if self.p0 is None :
            old_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            old_frame = cv2.resize(old_frame, (640, 480)) 

            if not os.path.exists('best_params.yaml') :
                cf = CostFunction(old_frame)
                self.best_params = cf.best_params
                with open('best_params.yaml', 'w') as f:
                    yaml.dump(self.best_params, f, default_flow_style=False)
            else :
                with open('best_params.yaml', 'r') as f:
                    self.best_params = yaml.safe_load(f)
            # self.best_params= {'kernel_size': 7, 'filter_type': 'gauss', 'thresholding_method': 'median'}

            self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            self.p0, self.lines, self.edges = find_new_points(old_frame, HOUGH_LINE_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, **self.best_params)

        else : 
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
            frame = cv2.resize(frame, (640, 480)) 

            window = frame.copy()
            self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try : 
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.frame_gray, self.p0, None, **lk_params)
                good_new = None
                if p1 is not None :
                    good_new = p1[st == 1].reshape(-1, 2)
                good_old = self.p0[st == 1].reshape(-1, 2)
                new_p, self.lines, self.edges = find_new_points(frame, HOUGH_LINE_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, **self.best_params)
                if good_new is not None and new_p is not None :
                    new_p = new_p.reshape(-1, 2) 
                    good_new = np.vstack((good_new, new_p))  
                elif new_p is not None : 
                    good_new = new_p.copy()
                else :
                    good_new = good_old.copy()

                valid_points = [] 

                for point in good_new:
                    if is_point_on_valid_line(point, self.lines):
                        valid_points.append(point)
                
                valid_points = np.array(valid_points)
                valid_points = remove_close_points(valid_points, POINT_DISTANCE_THRESHOLD)
                self.p0 = np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)

            except Exception as e :
                        print(traceback.format_exc())

            finally : 
        
                if self.lines is not None :
                    for line in self.lines :
                        x1, y1, x2, y2 = line
                        # angle = get_angle(x1, y1, x2, y2)      

                        # """only vertical lines"""
                        # if abs(angle) > 80 and abs(angle) < 100 :  # vertical line 
                        cv2.line(window, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.circle(window, (x1, y1), 5, (0, 255, 0), -1)
                        cv2.circle(window, (x2, y2), 5, (0, 255, 0), -1)

                cv2.imshow('Frame with Optical Flow', cv2.hconcat([window, cv2.cvtColor(self.edges,cv2.COLOR_GRAY2BGR)]))
                out.write(window)
                cv2.waitKey(2)

                self.old_gray = self.frame_gray.copy()