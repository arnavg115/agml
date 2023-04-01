import numpy as np

def display_mnist_image(image):
    """
    Image needs to be grayscale and normalized to [0,1]
    """
    chars = np.asarray(list(' .,:irs@9B&#'))
    scaled_image = (image.astype(float)) * (chars.size - 1)
    ascii_image = chars[scaled_image.astype(int)]
    print('\n'.join(''.join(row) for row in ascii_image))