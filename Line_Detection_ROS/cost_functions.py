import cv2, numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score
import timeit
from utils import *
from scipy.spatial.distance import cdist
from scipy.ndimage import sobel

class CostFunction :
    """input first frame"""
    def __init__(self, first_frame) :
        self.best_params = self.searching_best_params(first_frame)

    """search best parameters with Grid search"""
    def searching_best_params(self, first_frame) :
        filter_type = ['gauss', 'median', ] 
        thresholding_method = ['median','otsu']
        kernel_sizes=[5,7,9]

        best_score = 0
        best_params = {}

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        for f, ksize, m in product(filter_type, kernel_sizes, thresholding_method) :
            rand_scores = 0  
            for i in range(10) :
                GT = self.edge_detection(gray, f, ksize, m)
                rand_img = self.transform_img(gray, seed=i)
                rand_edge = self.edge_detection(rand_img, f, ksize, m)
                rand_scores += self.compute_weighted_metrics(gray, GT, rand_edge, f, ksize, m)

            rand_scores = rand_scores / 10 

            if rand_scores > best_score:
                best_score = rand_scores
                best_params = {"kernel_size" : ksize, "filter_type" : f,
                            'thresholding_method' : m}
        print('=' * 25)
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        return best_params

    """computing weighted metrics"""
    def compute_weighted_metrics(self, gray, ground_truth_image, edge_image, filter_type, kernel_size, thresholding_method):
        # f1, mcr, essim, continuity, fom, processing time, 
        f1_score, mcr = self.compute_f1_and_MCR(ground_truth_image, edge_image)
        essim = self.compute_essim(ground_truth_image, edge_image)
        continuity = self.compute_continuity(ground_truth_image)
        doe = self.compute_doe(ground_truth_image, edge_image)
        # processing_time = self.compute_processing_time(gray, filter_type, kernel_size, thresholding_method)
        return (f1_score * 0.15) + (mcr * 0.1) + (doe * 0.25) + (essim * 0.25) + (continuity * 0.25) # - (processing_time * 0.1) 

    """f1 score and misclassification rate(MCR)"""
    def compute_f1_and_MCR(self, ground_truth_image, edge_image) :
        TP = np.sum((ground_truth_image == 255) & (edge_image == 255))
        FP = np.sum((ground_truth_image == 0) & (edge_image == 255))
        FN = np.sum((ground_truth_image == 255) & (edge_image == 0))
        TN = np.sum((ground_truth_image == 0) & (edge_image == 0))
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        
        # harmonic mean of recall and precision 
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        mcr = (FP + FN) / (TP + TN + FP + FN) 
        return f1_score, 1 - mcr
    
    def compute_doe(self, gt_image, detected_image, alpha=0.1, beta=0.1):
        """compute displacement of edge images"""
        gt_direction = self.compute_gradient_directions(gt_image)
        det_direction = self.compute_gradient_directions(detected_image)
        
        gt_edges = np.argwhere(gt_image == 255)
        det_edges = np.argwhere(detected_image == 255)

        N_max = max(len(gt_edges), len(det_edges))
        doe_sum = 0
        
        if gt_edges.size > 0 and det_edges.size > 0:
            distances = cdist(det_edges, gt_edges, 'euclidean')
            for i, det_edge in enumerate(det_edges):
                min_idx = np.argmin(distances[i])
                gt_edge = gt_edges[min_idx]
                direction_diff = np.abs(gt_direction[gt_edge[0], gt_edge[1]] - det_direction[det_edge[0], det_edge[1]])
                direction_diff = np.mod(direction_diff + np.pi, 2 * np.pi) - np.pi
                doe_sum += 1 / (1 + alpha * distances[i][min_idx] ** 2 + beta * direction_diff ** 2)
        
        doe = doe_sum / N_max
        return doe

    """compute essim"""
    def compute_essim(self, ground_truth_image, edge_image):
        luminance1, luminance2 = self.compute_luminance(ground_truth_image, edge_image)
        contrast1, contrast2 = self.compute_contrast(ground_truth_image, edge_image)

        l = (2 * luminance1 * luminance2 + 0.01) / (luminance1**2 + luminance2**2 + 0.01)
        c = (2 * contrast1 * contrast2 + 0.03) / (contrast1**2 + contrast2**2 + 0.03)
        e = (np.cov(ground_truth_image.ravel(), edge_image.ravel())[0, 1] + 0.01) / (contrast1 * contrast2 + 0.01)

        return l * 0.33 + c * 0.33 + e * 0.33

    """compute processing time"""
    def compute_processing_time(self, gray, filter_type, kernel_size, thresholding_method) :
        execution_time = timeit.timeit(lambda: self.edge_detection(gray, filter_type, kernel_size, thresholding_method),
                                        number=60)
        return execution_time

    """compute continuity"""
    def compute_continuity(self, edge_image) : 
        edge_seg, edgels = self.find_edges_and_segments(edge_image)
        epsilon = 1e-10
        continuity = len(edgels) / (len(edge_seg) + epsilon)
        return continuity

    def compute_gradient_directions(self, image):
        """compute pixel gradient direction with sobel filter"""
        dx = sobel(image, axis=1, mode='constant')
        dy = sobel(image, axis=0, mode='constant')
        directions = np.arctan2(dy, dx)
        return directions


    """compute luminance of two images"""
    def compute_luminance(self, img1, img2):
        return np.mean(img1), np.mean(img2)

    """compute contrast of two images"""
    def compute_contrast(self, img1, img2):
        return np.std(img1), np.std(img2)

    """find edgel and edge lines to compute continuity"""
    def find_edges_and_segments(self, binary_image, threshold=20):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        edge_segments = contours  
        
        edge_lines = []
        for contour in contours:
            if len(contour) > threshold :  
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                edge_lines.append(approx)
        
        return edge_segments, edge_lines

    """edge detection with certain params (median, gauss filter only)"""
    def edge_detection(self, image, filter_type, kernel_size, thresholding_method):
        if len(image.shape) == 3 :
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else :
            gray = image.copy()
        
        if filter_type == 'gauss' : 
            blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        elif filter_type == 'median' : 
            blur = cv2.medianBlur(gray, kernel_size)
        else :
            blur = gray.copy()

        if thresholding_method == 'median' :
            edge = auto_canny(blur)
        elif thresholding_method == 'otsu' : 
            edge = auto_canny_otsu(blur)
        else : 
            edge = canny(blur)

        return edge


    """random transformation with brightness, contrast"""
    def transform_img(self, img, seed) : 
        range1 = np.arange(-50, 50, 5) 
        np.random.seed(seed)
        factor1 = np.random.choice(range1)
        range2 = np.arange(0.2, 1.6, 0.1)
        np.random.seed(seed+1)
        factor2 = np.random.choice(range2)
        
        return  cv2.convertScaleAbs(img, alpha=factor2, beta=factor1)
