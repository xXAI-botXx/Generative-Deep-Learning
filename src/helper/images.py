

import cv2

def imshow(img, title=None, image_width=10, axis=False,
           color_space="RGB", cols=1, save_to=None,
           hspace=0.2, wspace=0.2,
           use_original_sytle=False, invert=False):
    """
    Visualizes one or multiple images.

    Image will be reshaped: [batch_size/images, width, height, channels]

    title can be None, str or a list of strings.
    """
    import cv2

    original_style = plt.rcParams.copy()

    img_shape = img.shape
    print(f"Got images with shape: {img_shape}")

    # tranform the image to the right form
    if len(img_shape) == 2:
        img = np.reshape(img, shape=(1, img.shape[0], img.shape[1], 1))
    elif len(img_shape) == 3:
        # check if multiple gray images or multiple images with channel
        if img.shape[2] < img.shape[0] and img.shape[1] == img.shape[2]:
            img = np.reshape(img, shape=(1, img.shape[0], img.shape[1], img.shape[3]))
        else:
            # there could be cases where this is wrong
            img = np.reshape(img, shape=(img.shape[0], img.shape[1], img.shape[3], 1))
        img = np.reshape(img, shape=(1, img.shape[0], img.shape[1], 1))
    elif len(img_shape) != 4:
        raise ValueError(f"Image(s) have wrong shape! Founded shape: {img.shape}.")

    print(f"Transformed shape to: {img_shape}")

    # invert images
    if invert:
        print("Invert images...")
        max_value = 2**(img.dtype.itemsize * 8) -1
        scaling_func = lambda x: max_value - x
        img = np.apply_along_axis(scaling_func, axis=0, arr=img)

    # Set visualization settings
    # aspect_ratio_width = img.shape[1] / img.shape[2]
    aspect_ratio = img.shape[2] / img.shape[1]

    n_images = img.shape[0]
    rows = n_images//cols + int(n_images % cols > 0)

    width = int(image_width * cols)
    height = int(image_width * rows * aspect_ratio)

    # set plt style
    if not use_original_sytle:
        plt_style = 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else np.random.choice(plt.style.available)
        plt.style.use(plt_style)
        print(f"Using '{plt_style}'' plotting style.")

    # plotting
    print(f"Making you a beautiful plot...")
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(width, height))
    ax = ax.ravel()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    if type(title) == str:
        fig.suptitle(title, fontsize=128, y=0.95)

    for idx in range(len(ax)):
        cur_ax = ax[idx]

        if idx >= len(img):
            cur_ax.axis("off")
            continue

        cur_img = img[idx]

        if color_space.lower() == "bgr":
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            cmap = None
        elif color_space.lower() == "rgb":
            cur_img = cur_img
            cmap = None
        elif color_space.lower() == "hsv":
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_HSV2RGB)
            cmap = None
        elif color_space.lower() in ["gray", "grey", "g"]:
            cur_img = cur_img
            cmap = "gray"

        cur_ax.imshow(cur_img, cmap=cmap)

        if type(title) in [list, tuple]:
            cur_ax.set_title(title[idx], fontsize=64)
        if axis == False:
            cur_ax.axis("off")

    if save_to:
        os.makedirs(os.path.split(save_to)[0], exist_ok=True)
        fig.savefig(save_to, dpi=300)

    plt.show()

    if not use_original_sytle:
        # reset to original plt style
        plt.rcParams.update(original_style)




def get_used_depth(img:np.ndarray):
    """
    Find the number of possible/used pixel values in an image.

    Only works if the pixel space is really used! 
    """
    max_value = img.max()
    if max_value - 1 <= 0:
        return 1
    elif max_value - 2**8-1 <= 0:
        return 8
    elif max_value - 2**16-1 <= 0:
        return 16
    elif max_value - 2**32-1 <= 0:
        return 32
    else:
        raise ValueError("The depth of the given image is not sure:", max_value)



def get_depth(img:np.ndarray):
    """
    Returns the depth of an image in bit.
    """
    return img.dtype.itemsize * 8



def change_bit_depth_with_scaling(image, new_bit_depth=None):
    old_dtype = image.dtype
    int_types = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    float_types = {16: np.float16, 32: np.float32, 64: np.float64}

    # old depth pixel space
    old_bit_depth = old_dtype.itemsize * 8
    old_min, old_max = (0, 2**old_bit_depth - 1) if np.issubdtype(old_dtype, np.integer) else (0.0, 1.0)
    print(f"Old bit depth: {old_bit_depth} bit")

    # new datatype
    if new_bit_depth is None:
        new_bit_depth = get_used_depth(image)
        print(f"Found a used depth space of {new_bit_depth} bit")

    if np.issubdtype(old_dtype, np.integer):
        new_dtype = int_types.get(new_bit_depth, None)
        new_min, new_max = 0, 2**new_bit_depth - 1
    elif np.issubdtype(old_dtype, np.floating):
        new_dtype = float_types.get(new_bit_depth, None)
        new_min, new_max = 0.0, 1.0
    else:
        raise ValueError("Unsupported dtype")

    if new_dtype is None:
        raise ValueError(f"Unsupported bit depth: {new_bit_depth}")

    # scaling and applying
    if new_dtype == old_dtype:
        print("No datatyp change done! But dat got scaled")
    else:
        print(f"Change and scaled from {old_dtype} ({old_bit_depth} bit) -> {new_dtype} ({new_bit_depth} bit)")
        
    norm_array = (image.astype(np.float32) - old_min) / (old_max - old_min)
    scaled_array = norm_array * (new_max - new_min) + new_min

    return scaled_array.astype(new_dtype)








