import numpy as np
import matplotlib.pyplot as plt
as_strided  = np.lib.stride_tricks.as_strided


def compute_disparity(img1, img2, blk = 4):

    h, w = img1.shape
    w_out = w - blk + 1
    h_out = h - blk + 1

    view_shape = (h_out, w_out, blk, blk)
    view_strides = (*(img1.strides), *(img1.strides))

    imblk1 = as_strided(img1, view_shape, view_strides)
    imblk2 = as_strided(img2, view_shape, view_strides)
    disparity = np.zeros((h_out, w_out))

    imblk1 = np.swapaxes(imblk1, 0, 1)

    for x in range(w_out):
        min_x = max(x - 60, 0)
        max_x = min(x + 60, w_out)
        diff = imblk1[min_x:max_x, :] - imblk2[:, x]
        ssd = np.sum(diff**2, (2, 3))
        # ssdn = np.zeros_like(ssd)
        # ssdn[:2:] = ss[]
        disparity[:, x] = np.abs(min_x + np.argmin(ssd, axis = 0) - x)

    disparity_map_int = np.uint8(disparity * 255 / np.max(disparity))
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    # plt.savefig('disparity_image_gray.png')

    return disparity, disparity_map_int