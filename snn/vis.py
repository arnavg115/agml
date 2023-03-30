import numpy as np

def display_mnist_image(image):
    chars = np.asarray(list(' .,:irs@9B&#'))
    scaled_image = (image.astype(float) / 255) * (chars.size - 1)
    ascii_image = chars[scaled_image.astype(int)]
    print('\n'.join(''.join(row) for row in ascii_image))