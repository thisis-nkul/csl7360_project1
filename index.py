import gradio as gr
import numpy as np
import cv2 as cv
import PIL
import os
import pandas as pd
import cv2
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image):
        pass

class HistogramFeatureExtractor(FeatureExtractor):
    def __init__(self, bins=8):
        self.bins = bins

    def extract_features(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [self.bins, self.bins, self.bins],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist


class MultiScaleHistogramFeatureExtractor(FeatureExtractor):
    def __init__(self, scales=[2, 4], bins=16):
        self.bins = bins
        self.scales = scales

    def _extract_features(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [self.bins, self.bins, self.bins],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def extract_features(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        f = None

        for s in self.scales:
            h, w, _ = image.shape
            image = cv2.resize(image, (h//s, w//s))
            sc_features = self._extract_features(image)

            f = np.concatenate([f, sc_features]) if f is not None else sc_features
        
        return f
        

class ColorMomentFeatureExtractor(FeatureExtractor):
    def extract_features(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]
        features = []
        for channel in [red_channel, green_channel, blue_channel]:
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-10)
            kurtosis = np.mean((channel - mean) ** 4) / (std ** 4 + 1e-10)
            features.extend([mean, std, skewness, kurtosis])
        return np.array(features)

class SimilarityMeasure(ABC):
    @abstractmethod
    def compute_similarity(self, features, query_feature):
        pass


class NormalizedCosineSim(SimilarityMeasure):
    def compute_similarity(self, features, query_feature):
        similarities = np.dot(features - features.mean(-1, keepdims=True), query_feature - query_feature.mean(-1)) / (
                np.linalg.norm(features, axis=1) * np.linalg.norm(query_feature)
        )
        return similarities

class DotProductSimilarity(SimilarityMeasure):
    def compute_similarity(self, features, query_feature):
        similarities = np.dot(features, query_feature) / (
                np.linalg.norm(features, axis=1) * np.linalg.norm(query_feature)
        )
        return similarities

class EuclideanDistanceSimilarity(SimilarityMeasure):
    def compute_similarity(self, features, query_feature):
        distances = np.linalg.norm(features - query_feature, axis=1)
        similarities = 1 / (distances + 1e-10)
        return similarities


class ImageRetriever:
    def __init__(self, feature_extractor, similarity_measure):
        self.feature_extractor = feature_extractor
        self.similarity_measure = similarity_measure

    def index_images(self, image_paths):
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            feature = self.feature_extractor.extract_features(image)
            features.append(feature)
        self.features = np.array(features)

    def search(self, query_image, top_k=5):
        query_feature = self.feature_extractor.extract_features(query_image)
        similarities = self.similarity_measure.compute_similarity(self.features, query_feature)
        indices = np.argsort(similarities)[::-1][:top_k]
        return indices

## Bag of Visual Words + Fisher Feature

from sklearn.cluster import KMeans
def BagOfVisualWords_SIFT(Gallery,vocab_size=50):
  BOVW_SIFT=[]
  # Load image from path
  for index, i in Gallery.iterrows():
    # Load image from path
    image = cv2.imread(i['File'])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create SIFT object
    sift = cv2.SIFT_create()
    # Find keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    for j in range(descriptors.shape[0]):
      BOVW_SIFT.append(descriptors[j])
    
  BOVW_quantizer = KMeans(n_clusters=vocab_size, random_state=0, n_init="auto").fit(BOVW_SIFT)

  return BOVW_quantizer

def BOVW_Quantized_Vector(image,BOVW_quantizer):
  n=BOVW_quantizer.cluster_centers_.shape[0]
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Create SIFT object
  sift = cv2.SIFT_create()
  # Find keypoints and descriptors
  keypoints, descriptors = sift.detectAndCompute(gray_image, None)
  output=np.zeros(n)
  for j in range(descriptors.shape[0]):
      output[BOVW_quantizer.predict([descriptors[j]])[0]]+=1

  return output/descriptors.shape[0]
  # return output

def create_BOVWQV_Gallery(Gallery,BOVW_quantizer):
  y=Gallery['Class']
  X=[]
  for index, i in Gallery.iterrows():
    image = cv2.imread(i['File'])
    X.append(BOVW_Quantized_Vector(image,BOVW_quantizer))
  return np.array(X),y

def computeFisherProjection(X, y, n_components):
    """
    X: data matrix of shape (n_samples, n_features)
    y: target vector of shape (n_samples,)
    n_components: number of components to keep
    
    returns: projection matrix of shape (n_features, n_components)
    """
    # Compute class means
    class_means = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])
    
    # Compute overall mean
    overall_mean = np.mean(X, axis=0)
    
    # Compute between-class scatter matrix
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for i, mean_vec in enumerate(class_means):
        ni = X[y==i,:].shape[0]
        mean_vec = mean_vec.reshape(X.shape[1], 1)  # make column vector
        overall_mean = overall_mean.reshape(X.shape[1], 1)  # make column vector
        Sb += ni * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        
    # Compute within-class scatter matrix
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i in np.unique(y):
        Xi = X[y == i]
        mean_vec = class_means[i]
        mean_vec = mean_vec.reshape(X.shape[1], 1)  # make column vector
        for x in Xi:
            x = x.reshape(X.shape[1], 1)  # make column vector
            Sw += (x - mean_vec).dot((x - mean_vec).T)
    
    # Compute eigenvalues and eigenvectors of (Sw^-1)Sb
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    # Sort eigenvectors by decreasing eigenvalues and select the first n_components
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    print(len(eig_pairs))
    eigvecs_sorted = np.array([eig_pairs[i][1] for i in range(n_components)])
    
    # Compute projection matrix
    projection_matrix = eigvecs_sorted.T
    
    return projection_matrix

class Global_SIFT(FeatureExtractor):
  def __init__(self,BOVW_quantizer,proj):
    self.BOVW_quantizer=BOVW_quantizer
    self.proj = proj

    pass
  def extract_features(self, image):

    # return np.dot(np.dot(self.proj,BOVW_Quantized_Vector(image,self.BOVW_quantizer)).T,self.BOVW_quantizer.cluster_centers_)
    return np.dot(self.proj,BOVW_Quantized_Vector(image,self.BOVW_quantizer))

import pickle
import matplotlib.pyplot as plt
import cv2

def show_image(image_path):
    # Read image from file
    
    # print(new_path)
    image = cv2.imread(image_path[1:])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def Retrive(image,mode):
    # Defining Feature Extractors and Similarity measures
    X_Gallery = pd.read_csv("out.csv")
    Global_SIFT_feature=np.load("Global_SIFT.npy")
    histFeature=np.load("hist_feature_extractor.npy")
    colorMomentFeature=np.load("color_moment_feature_extractor.npy")
    KMeans50Feature= pickle.load(open("KMeans50.pkl", "rb"))
    proj=np.load("ProjectionMatrix.npy")
    MultiscaleFeatures=np.load("MultiScale.npy")
    multi_scale_hist_feature_extractor = MultiScaleHistogramFeatureExtractor()
    globalSiftExtractor=Global_SIFT(KMeans50Feature,proj)
    hist_feature_extractor = HistogramFeatureExtractor()
    color_moment_feature_extractor = ColorMomentFeatureExtractor()
    dot_product_similarity_measure = DotProductSimilarity()
    euclidean_distance_similarity_measure = EuclideanDistanceSimilarity()
    normalized_cosine_sim = NormalizedCosineSim()
    output = [image]*5
    if(mode == "MultiScale Histogram"):
        output=[]
        retriever = ImageRetriever(multi_scale_hist_feature_extractor, normalized_cosine_sim)
        retriever.features = MultiscaleFeatures
        # Searching for similar images using the current feature extractor and similarity measure
        similar_image_indices = retriever.search(image, 5)
        for index in similar_image_indices:
            output.append(show_image(X_Gallery.iloc[index][0]))
    if(mode == "BOVW-Fisher"):
        output=[]
        retriever = ImageRetriever(globalSiftExtractor, euclidean_distance_similarity_measure)
        retriever.features = Global_SIFT_feature

        # Searching for similar images using the current feature extractor and similarity measure
        similar_image_indices = retriever.search(image, 5)
        for index in similar_image_indices:
            output.append(show_image(X_Gallery.iloc[index][0]))
        
    if(mode == "Histogram"):
        output=[]
        retriever = ImageRetriever(hist_feature_extractor, dot_product_similarity_measure)
        retriever.features = histFeature

        # Searching for similar images using the current feature extractor and similarity measure
        similar_image_indices = retriever.search(image, 5)
        for index in similar_image_indices:
            output.append(show_image(X_Gallery.iloc[index][0]))
    if(mode == "ColorMoment"):
        output=[]
        retriever = ImageRetriever(color_moment_feature_extractor, euclidean_distance_similarity_measure)
        retriever.features = colorMomentFeature

        # Searching for similar images using the current feature extractor and similarity measure
        similar_image_indices = retriever.search(image, 5)
        for index in similar_image_indices:
            output.append(show_image(X_Gallery.iloc[index][0]))

    return output


top_k = 5

demo = gr.Interface(fn=Retrive, inputs=["image",gr.Radio(["MultiScale Histogram", "BOVW-Fisher", "Histogram","ColorMoment"], label="Method", info="Which Feature to use")], outputs=["image","image","image","image","image"])

demo.launch()