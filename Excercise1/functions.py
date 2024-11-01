import numpy as np

# filter_size = len of x-edge
def padding(img, p_size):
    row, col = img.shape
    top = np.repeat([img[0]], repeats=p_size, axis=0) # top padding
    bottom = np.repeat([img[-1]], repeats=p_size, axis=0) # bottom padding
    img = np.vstack([top, img, bottom]) 
    img = np.insert(img, [col]*p_size, img[:,[col-1]], axis=1) # right padding
    img = np.insert(img, [1]*p_size, img[:,[0]], axis=1) # left padding
    return img


# (a) Implement two functions that return cross-correlation between an image and a 1D/2D kernel
def cross_correlation_1d(img, kernel=None):
    processed_img = []
    filter_size = kernel.shape[0] # filter의 길이

    # (c) You can assume that the all kernels are odd sized along both dimensions.
    p_size = filter_size // 2 # padding의 크기
    padded_img = padding(img, p_size=p_size) # padding image
    row, col = padded_img.shape

    # (b) Your function cross_correlation_1d should distinguish between vertical and horizontal
    # kernels based on the shape of the given kernel.
    for i in range(p_size, row-p_size):
        temp  = []
        for j in range(p_size, col-p_size):
            if kernel.ndim == 1: temp.append(np.dot(kernel, padded_img[i, j-p_size:j+p_size+1]))
            else: temp.append(np.dot(np.ravel(kernel), padded_img[i-p_size:i+p_size+1, j]))
        processed_img.append(temp)
    return np.array(processed_img)


def cross_correlation_2d(img, kernel=None):
    processed_img = []
    filter_size = kernel.shape[0] # filter의 한 변의 길이 (MxM Kernel이라고 할 때 ..)

    # (c) You can assume that the all kernels are odd sized along both dimensions.
    p_size = filter_size // 2 # padding의 크기
    padded_img = padding(img, p_size=p_size) # padding image
    row, col = padded_img.shape

    for i in range(p_size, row-p_size):
        temp = []
        for j in range(p_size, col-p_size):
            temp.append(np.dot(kernel.ravel(), padded_img[i-p_size:i+p_size+1, j-p_size:j+p_size+1].ravel()))
        processed_img.append(temp)
    return np.array(processed_img)


def gaussian(i, j=0, sigma=1):
    return (1/(2*np.pi*sigma**2))*np.exp(-1*((i**2)+(j**2))/(2*sigma**2))

def get_gaussian_filter_1d(size, sigma=1):
    # (b) You can assume that size is an odd number.
    k_size = size // 2
    kernel = np.linspace(-k_size, k_size, size) # 1D kernel의 사이즈를 기준으로 x-domain 생성
    kernel = gaussian(kernel, j=0, sigma=sigma)
    kernel = kernel / kernel.sum()
    return kernel

def get_gaussian_filter_2d(size, sigma=1):
    # (b) You can assume that size is an odd number.
    k_size = size // 2 # size : 한 변의 길이
    a = np.linspace(-k_size, k_size, size)
    b = np.linspace(-k_size, k_size, size)
    x, y = np.meshgrid(a, b)
    kernel = gaussian(x, y, sigma=sigma)
    kernel = kernel / kernel.sum()
    return kernel