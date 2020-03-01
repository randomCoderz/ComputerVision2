import cv2 as cv
import numpy as np


# Pass in grayscale image, min and max threshold
def sobel(ig):
    # Apply Gaussian Blur
    img1 = cv.GaussianBlur(ig, (5,5), 1)
    # Apply Sobel
    sobelx = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=5)
    g = np.hypot(sobelx, sobely)
    # Get in degrees
    theta = np.degrees(np.arctan2(sobelx, sobely))
    return g, theta


def non_max_suppression(ig, di, g):
    z_image = np.zeros(ig.shape)
    di[di < 0] += 180

    # Iterate
    for i in range(z_image.shape[0]):
        for j in range(z_image.shape[1]):
            try:
                q = 255
                r = 255

                # 0 degrees
                if (0 <= di[i, j] < 22.5) or (157.5 <= di[i,j] <= 180):
                    q = g[i, j+1]
                    r = g[i, j-1]
                # 45 degrees
                elif 22.5 <= di[i, j] < 67.5:
                    q = g[i+1, j-1]
                    r = g[i, j-1]
                # 90 degrees
                elif 67.5 <= di[i, j] < 112.5:
                    q = g[i+1, j-1]
                    r = g[i-1, j+1]
                # 135 degrees
                elif 112.5 <= di[i, j] < 157.5:
                    q = g[i-1, j-1]
                    r = g[i+1, j+1]

                if(g[i,j] >= q) and(g[i,j] >= r):
                    z_image[i,j] = g[i,j]
                else:
                    z_image[i,j] = 0
            except IndexError:
                pass
    return z_image


def d_threshold(g_mag, l_ratio, h_ratio):
    high_threshold = g_mag.max() * h_ratio
    low_threshold = l_ratio * high_threshold
    thres_matrix = np.zeros(g_mag.shape)
    strong_pix, weak_pix = 255, 20

    for i in range(g_mag.shape[0]):
        for j in range(g_mag.shape[1]):
            current = g_mag[i][j]
            if current >= high_threshold:
                thres_matrix[i][j] = strong_pix
            elif current < low_threshold:
                thres_matrix[i][j] = 0
            else:
                thres_matrix[i][j] = weak_pix
    return thres_matrix, strong_pix, weak_pix


def hysteresis(ig, strong, weak):
    h_mat = ig.shape
    for i in range(h_mat[0]):
        for j in range(h_mat[1]):
            if ig[i, j] == weak:
                try:
                    if ig[i - 1, j + 1] == strong or ig[i, j + 1] == strong or ig[i + 1, j + 1] == strong:
                        ig[i, j] = strong
                    elif ig[i - 1, j] == strong or ig[i+1, j] == strong:
                        ig[i, j] = strong
                    elif ig[i-1, j-1] == strong or ig[i, j-1] == strong or ig[i+1, j-1] == strong:
                        ig[i, j] = strong
                    else:
                        ig[i, j] = 0
                except IndexError:
                    pass
    return ig


def manual_canny(ig, l_ratio, h_ratio):
    g, theta = sobel(ig)
    nonmax = non_max_suppression(ig, theta, g)
    matr, strong_pix, weak_pix = d_threshold(nonmax, l_ratio, h_ratio)
    final = hysteresis(matr, strong_pix, weak_pix)
    return final


# Pass in grayscale image, min and max threshold
def open_cv_canny(ig, mxi, mxa):
    return cv.Canny(ig, mxi, mxa)


def open_cv_Laplacian(ig, val):
    return cv.Laplacian(ig, val)


if __name__ == "__main__":
    # Define the image and make it grayscaled
    img = cv.imread('lambo.jpg', 0)
    thres_low = 0.05
    thres_high = 0.08
    mc = manual_canny(img, thres_low, thres_high)
    cv.imshow('MC', mc)
    cv.imshow('OpenCV Canny Image', open_cv_canny(img, thres_low, thres_high))
    cv.imshow('OpenCV Laplacian', open_cv_Laplacian(img, 0))
    cv.waitKey()
