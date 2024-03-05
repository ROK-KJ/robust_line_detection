from find_feature_points import LinePointTracker
import cv2
import rospy

def run() :
    rospy.init_node('LinePointTracker', anonymous=True)
    tracker = LinePointTracker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__' :
    run()
   
