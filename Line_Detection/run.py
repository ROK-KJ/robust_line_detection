from cost_functions import CostFunction
import argparse
from find_feature_points import tracking
import cv2

parser = argparse.ArgumentParser(description='Robust Line and Point Detection / Tracking')
parser.add_argument('--input', default='0', type=str, help="Input Video Name or Camera Number")
args = parser.parse_args()

def run(video_name) :
    if video_name.isnumeric() :
        video_name = int(video_name)
    cap = cv2.VideoCapture(video_name)
    _, frame = cap.read() 
    cf = CostFunction(frame)
    print(cf.best_params)

    tracking(cap, **cf.best_params)



if __name__ == '__main__' :
    run(args.input)
   
