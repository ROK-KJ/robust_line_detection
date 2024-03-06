import cv2, numpy as np

"""return new points, lines, edge image (simplified)"""
def find_new_points(frame, hough_threshold, min_line_length, max_line_gap, **params) : 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = run_best_edge_detector(gray, **params)
    simplified = simplify_edges(edges)
    lines = cv2.HoughLinesP(simplified, 1, np.pi/180, hough_threshold, minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
    if lines is not None :
        points = np.reshape(lines, (-1, 2))
        new_points = points.reshape(-1, 1, 2).astype(np.float32)
    else :
        new_points, lines = None, None
    return new_points, lines, simplified 

def calculate_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')  # Avoid division by zero
    return (y2 - y1) / (x2 - x1)

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
    if len(contours) < 15 : # not simplified
        simplified_edges = edge_image.copy()
    else : 
        count_threshold = np.percentile([len(c) for c in contours], 25)
        length_threshold = np.percentile([cv2.arcLength(c, True) for c in contours], 25)
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
