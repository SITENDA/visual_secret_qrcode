from PIL import Image
import numpy as np

def encode_color_shares(qr_code):
    qr_arr = np.array(qr_code)

    height, width = qr_arr.shape

    share1 = np.zeros((height, width, 3), dtype=np.uint8)
    share2 = np.zeros((height, width, 3), dtype=np.uint8)
    share3 = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Check the value of each pixel in the QR code
            value = qr_arr[i, j]

            # Determine the color values for each share based on the pixel value
            if value == 0:
                share1[i, j] = [255, 0, 0]  # Blue in share1
                share2[i, j] = [0, 255, 0]  # Green in share2
                share3[i, j] = [0, 0, 255]  # Red in share3
            else:
                share1[i, j] = [0, 0, 255]  # Red in share1
                share2[i, j] = [255, 0, 0]  # Blue in share2
                share3[i, j] = [0, 255, 0]  # Green in share3

    share1_img = Image.fromarray(share1, 'RGB')
    share2_img = Image.fromarray(share2, 'RGB')
    share3_img = Image.fromarray(share3, 'RGB')

    return share1_img, share2_img, share3_img

# Example usage
qr_code = Image.open("qr_code.png")  # Replace "qr_code.png" with the filename of your QR code image

share1, share2, share3 = encode_color_shares(qr_code)

# Save the color shares as image files or display them
share1.show()
share2.show()
share3.show()
