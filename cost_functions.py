import cv2, numpy as np
from itertools import product
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as compare_ssim
import timeit
from find_feature_points import auto_canny, auto_canny_otsu, canny

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
        essim = self.compute_essim(ground_truth_image, edge_image, filter_type, kernel_size, thresholding_method)
        continuity = self.compute_continuity(edge_image)
        fom = self.compute_fom(ground_truth_image, edge_image)
        processing_time = self.compute_processing_time(gray, filter_type, kernel_size, thresholding_method)
        # print(f1_score, mcr, essim, continuity, fom, processing_time)
        # return f1_score * 0.4 + essim * 0.35 + continuity * 0.35 - processing_time * 0.1
        return (f1_score * 0.2) + (mcr * 0.1) + (essim * 0.3) + (continuity * 0.3) - (processing_time * 0.1) 
        

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
    
    """compute FOM (form of merit)""" 
    def compute_fom(self, ground_truth_image, edge_image, alpha=0.1) : 
        N_g = len(ground_truth_image)
        N_d = len(edge_image)

        fom_sum = 0
        for d in edge_image : 
            distances = np.sqrt(np.sum((ground_truth_image - d) ** 2, axis=1))
            min_distance = np.min(distances)
            fom_sum += 1 / (1 + alpha * min_distance ** 2)
        return fom_sum / N_g

    """compute essim"""
    def compute_essim(self, ground_truth_image, edge_image, filter_type, ksize, thresholding_method):
        luminance1, luminance2 = self.compute_luminance(ground_truth_image, edge_image)
        contrast1, contrast2 = self.compute_contrast(ground_truth_image, edge_image)
        edges1, edges2 = self.edge_detection(ground_truth_image, filter_type, ksize, thresholding_method), self.edge_detection(edge_image, filter_type, ksize, thresholding_method)

        l = (2 * luminance1 * luminance2 + 0.01) / (luminance1**2 + luminance2**2 + 0.01)
        c = (2 * contrast1 * contrast2 + 0.03) / (contrast1**2 + contrast2**2 + 0.03)
        e = compare_ssim(edges1, edges2)

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

    """compute luminance of two images"""
    def compute_luminance(self, img1, img2):
        return np.mean(img1), np.mean(img2)

    """compute contrast of two images"""
    def compute_contrast(self, img1, img2):
        return np.std(img1), np.std(img2)


    """find edgel and edge lines to compute continuity"""
    def find_edges_and_segments(self, binary_image, threshold=15):
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
        else : 
            blur = cv2.medianBlur(gray, kernel_size)
            
        if thresholding_method == 'median' :
            edge = auto_canny(blur)
        elif thresholding_method == 'otsu' : 
            edge = auto_canny_otsu(blur)
        else : 
            edge = canny(blur)

        return edge


    """random transformation with brightness, contrast"""
    def transform_img(self, img, seed) : 
        v = np.arange(0.2, 2.1, 0.1)
        np.random.seed(seed)
        factor1 = np.random.choice(v)
        np.random.seed(seed+1)
        factor2 = np.random.choice(v)
        
        img = np.array(255 * (img / 255) ** factor1, dtype='uint8')
        return  cv2.multiply(img, factor2)


if __name__ == '__main__' :
    cap = cv2.VideoCapture("./test.mp4")
    _, first_frame = cap.read() 
    cf = CostFunction(first_frame)

    print(cf.best_params)