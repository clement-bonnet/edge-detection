import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.lib.function_base import quantile
from scipy.signal import convolve2d


GX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
GY = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

class SobelDetector:
    def __init__(self, img_path):
        self.edges = None
        self.img = None
        self.entropy = None
        self.process_img(img_path)

    def process_img(self, img_path):
        img = np.array(Image.open(img_path))
        self.img = img
        img_x = convolve2d(img, GX, mode="same")
        img_y = convolve2d(img, GY, mode="same")
        edges = np.sqrt(np.square(img_x) + np.square(img_y))
        edges = edges / edges.max()
        self.edges = edges
    
    def detect_edges(self, threshold):
        if self.edges is None:
            raise AttributeError(
                "No edges computed. Please run process_img first.")
        edges_detected = (self.edges >= threshold)
        return edges_detected
    
    def detect_edges_quantile(self, q):
        if self.edges is None:
            raise AttributeError(
                "No edges computed. Please run process_img first.")
        quantile_threshold = np.quantile(self.edges, q)
        edges_detected = (self.edges >= quantile_threshold)
        return edges_detected

    def detect_edges_entropy(self, N=5, fn="1-x"):
        if self.edges is None:
            raise AttributeError(
                "No edges computed. Please run process_img first.")
        self.compute_entropy_image(N=N, show=False, normalize=True)
        if fn == "1-x":
            entropy_thresholds = 1 - self.entropy
        elif fn == "exp(-x)":
            entropy_thresholds =np.exp( - self.entropy)
        elif fn == "x":
            entropy_thresholds = self.entropy
        elif fn == "(1-x)^2":
            entropy_thresholds = np.square(1 - self.entropy)
        elif fn == "sqrt(1-x)":
            entropy_thresholds = np.sqrt(1 - self.entropy)
        elif fn == "0.1/x":
            entropy_thresholds = 1/(10*self.entropy)
        else:
            raise ValueError("fn '{}' not valid".format(fn))
        edges_detected = self.edges >= entropy_thresholds
        return edges_detected
    
    def compute_entropy_threshold(self, threshold):
        edges_detected = self.detect_edges(threshold)
        edges_0 = edges_detected == False
        edges_1 = edges_detected == True
        p_0 = edges_0.sum()/len(edges_detected.reshape(-1))
        p_1 = edges_1.sum()/len(edges_detected.reshape(-1))
        entropy_0 = p_0*np.log(p_0) if p_0 != 0 else 0
        entropy_1 = (1-p_1)*np.log(1-p_1) if 1-p_1 != 0 else 0
        entropy = -(entropy_0 + entropy_1)
        return entropy
    
    def compute_entropy_image(self, N=5, show=True, normalize=True):
        def get_entropy(signal):
            lensig = signal.size
            symset = list(set(signal))
            propab = np.array([np.size(signal[signal==k])/lensig for k in symset])
            ent = -np.sum(propab*np.log(propab))
            return ent
        (h,w) = self.img.shape
        entropy = np.zeros((h,w))
        for row in range(h):
            for col in range(w):
                Lx=np.max([0,col-N])
                Ux=np.min([w,col+N])
                Ly=np.max([0,row-N])
                Uy=np.min([h,row+N])
                region=self.img[Ly:Uy,Lx:Ux].flatten()
                entropy[row,col]=get_entropy(region)
        if normalize:
            entropy = (entropy - entropy.min())/(entropy.max() - entropy.min())
        self.entropy = entropy
        if show:
            plt.imshow(entropy, cmap=plt.cm.jet)
            plt.colorbar()
            plt.axis("off")
    
    def compute_margin(self, threshold):
        edges_detected = self.detect_edges(threshold)
        mask_0 = edges_detected==0
        mask_1 = edges_detected==1
        margin = np.sum(self.edges[mask_0]) +\
            np.sum((1 - self.edges[mask_1]))
        return margin

    def compute_squared_margin(self, threshold):
        edges_detected = self.detect_edges(threshold)
        mask_0 = edges_detected==0
        mask_1 = edges_detected==1
        squared_margin = np.sum(np.square(self.edges[mask_0])) +\
            np.sum(np.square(1 - self.edges[mask_1]))
        return squared_margin
    
    def compute_other_squared_margin(self, threshold):
        edges_detected = self.detect_edges(threshold)
        mask_0 = edges_detected==0
        mask_1 = edges_detected==1
        print(mask_0.sum())
        print(mask_1.sum())
        squared_margin = np.sum(np.square(self.edges[mask_0] - threshold)) +\
            np.sum(np.square(self.edges[mask_1] - threshold))
        return squared_margin

    def show_img(self, cmap=None):
        plt.imshow(self.img, cmap)
        plt.axis("off")

    def show_edges(self, cmap=None):
        plt.imshow(self.edges, cmap)
        plt.axis("off")
    
    def show_detected_edges(self, threshold, cmap=None):
        plt.imshow(self.detect_edges(threshold), cmap)
        plt.axis("off")

    def show_detected_edges_quantile(self, q, cmap=None):
        plt.imshow(self.detect_edges_quantile(q), cmap)
        plt.axis("off")
    
    def show_detected_edges_entropy(self, cmap=None, N=5, fn="1-x"):
        plt.imshow(self.detect_edges_entropy(N=N, fn=fn), cmap)
        plt.axis("off")


if __name__ == "__main__":
    thresholds = [10,50,100,150,200,250]
    sobel = SobelDetector()
    sobel.process_img("lena.jpg")
    for threshold in thresholds:
        edges = sobel.detect_edges(threshold)
        plt.imshow(edges)
