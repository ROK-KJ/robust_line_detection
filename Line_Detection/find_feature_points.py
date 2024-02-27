import cv2, numpy as np
import traceback 

POINT_DISTANCE_THRESHOLD=25 
HOUGH_LINE_THRESHOLD=100
MIN_LINE_LENGTH=120
MAX_LINE_GAP=100
lk_params = dict(winSize=(21, 21), maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                           10, 0.03))

def tracking(cap, **best_params) :
    """tracking initiation"""
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0, lines, edges = find_new_points(old_frame, HOUGH_LINE_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, **best_params)

    while True:
        ret, frame = cap.read()
        # start = time.time()
        if not ret:
            break
        window = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try : 
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            if p1 is not None :
                good_new = p1[st == 1].reshape(-1, 2)
            good_old = p0[st == 1].reshape(-1, 2)
            new_p, lines, edges = find_new_points(frame, HOUGH_LINE_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, **best_params)
            if len(new_p) > 0:
                new_p = new_p.reshape(-1, 2) 
                good_new = np.vstack((good_new, new_p))  
            else : 
                good_new = new_p.copy()
            # inside_frame = (good_new[:, 0] > 0) & (good_new[:, 0] < frame.shape[1]) & \
            #             (good_new[:, 1] > 0) & (good_new[:, 1] < frame.shape[0])
            # good_new = good_new[inside_frame]

            valid_points = [] 

            for point in good_new:
                if is_point_on_valid_line(point, lines):
                    valid_points.append(point)
            
            valid_points = np.array(valid_points)
            valid_points = remove_close_points(valid_points, POINT_DISTANCE_THRESHOLD)
            p0 = np.array(valid_points, dtype=np.float32).reshape(-1, 1, 2)

        except Exception as e :
            print(traceback.format_exc())
        
        finally : 
            
            if lines is not None :
                for line in lines :
                    x1, y1, x2, y2 = line[0]
                    angle = get_angle(x1, y1, x2, y2)               
                    """only vertical lines"""
                    if abs(angle) > 80 and abs(angle) < 100 :  # vertical line 
                        cv2.line(window, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.circle(window, (x1, y1), 5, (0, 255, 0), -1)
                        cv2.circle(window, (x2, y2), 5, (0, 255, 0), -1)
                        
                    else :
                        cv2.line(window, (x1, y1), (x2, y2), (255,0, 0), 1)   
            
            cv2.imshow('Frame with Optical Flow', cv2.hconcat([window, cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)]))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    #         print(f"computing time : {time.time() - start : .2f}")
            old_gray = frame_gray.copy()
        
    cv2.destroyAllWindows()
    cap.release()


"""return new points, lines, edge image (simplified)"""
def find_new_points(frame, hough_threshold, min_line_length, max_line_gap, **params) : 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = run_best_edge_detector(gray, **params)
    simplified = simplify_edges(edges)
    lines = cv2.HoughLinesP(simplified, 1, np.pi/180, hough_threshold, minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
    points = np.reshape(lines, (-1, 2))
    new_points = points.reshape(-1, 1, 2).astype(np.float32)
    return new_points, lines, simplified 

"""edge detecting with best params"""
def run_best_edge_detector(image, **params) :
    if params :
        ftype = params['filter_type']
        ksize = params['kernel_size']
        thresholding_method = params['thresholding_method']
        if ftype == 'gauss' :
            blur = cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif ftype == 'median' : 
            blur = cv2.medianBlur(image, ksize)
        else :
            blur = image.copy()
        
        if thresholding_method == 'median' : 
            edge_image = auto_canny(blur)
        else :
            edge_image = auto_canny_otsu(blur)
    else :
        edge_image = canny(image)
    return edge_image 


"""edge simplifier"""
def simplify_edges(edge_image) : 
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count_threshold = np.percentile([len(c) for c in contours], 50)
    length_threshold = np.percentile([cv2.arcLength(c, True) for c in contours], 50)
    contour_list = [c for c in contours if len(c) > count_threshold and cv2.arcLength(c, True) > length_threshold]         
    simplified_edges = np.zeros(edge_image.shape, dtype=np.uint8)
    cv2.drawContours(simplified_edges, contour_list, -1, (255, 255, 255), 1)
    return simplified_edges


def remove_close_points(points, threshold):
    valid_points = []
    for i, point in enumerate(points):
        too_close = False
        for j, other_point in enumerate(points):
            if i != j and distance_between_points(point, other_point) < threshold:
                too_close = True
                break
        if not too_close:
            valid_points.append(point)
    return np.array(valid_points)


def is_point_on_valid_line(point, lines, threshold=25): # threshold=10
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = cv2.pointPolygonTest(np.array([[x1, y1], [x2, y2]]), (point[0], point[1]), True)
        if abs(dist) < threshold:
            return True
    return False


def get_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def distance_between_points(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


"""Edge Detecting Method (Canny thresholding)"""
def auto_canny(img, sigma=0.33) : 
    if len(img.shape) == 3 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(img)
    lower = int(max(0, 1.0 - sigma)*v)
    upper = int(min(255, (1.0 + sigma)*v ))
    return cv2.Canny(img, lower, upper)

def auto_canny_otsu(image):
    if len(image.shape) == 3 :
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5 * high_thresh
    return cv2.Canny(image, low_thresh, high_thresh)

def canny(img) : 
    if len(img.shape) == 3 :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, 100, 200)