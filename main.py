import qrcode
from PIL import Image
import numpy as np
import random

from secret_qr_code import generate_secret_qr_code, extract_data_encoding_type, extract_bit_stream, get_qr_code_version, \
    convert_to_codewords, QR_CODE_VERSION_INFO, assemble_blocks, get_data_blocks, perform_reed_solomon_encoding


def create_color_shares_new(input_image):
    # Convert the input image to grayscale
    gray_image = input_image.convert("L")
    arr = np.array(gray_image)

    # Get the dimensions of the grayscale image
    height, width = arr.shape

    # Initialize color shares as empty images
    share1 = np.zeros((height, width, 3), dtype=np.uint8)
    share2 = np.zeros((height, width, 3), dtype=np.uint8)
    share3 = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate through each pixel/module in the grayscale image
    for i in range(height):
        for j in range(width):
            grayscale_value = arr[i, j]

            # Determine colors for each share based on grayscale value
            if grayscale_value < 85:
                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Blue, Green, Red
            elif grayscale_value < 170:
                colors = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]  # Red, Blue, Green
            else:
                colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0]]  # Green, Red, Blue

            # Shuffle the colors randomly
            np.random.shuffle(colors)

            # Assign colors to each share
            share1[i, j] = colors[0]
            share2[i, j] = colors[1]
            share3[i, j] = colors[2]

    # Create PIL images from the color shares
    share1_img = Image.fromarray(share1, 'RGB')
    share2_img = Image.fromarray(share2, 'RGB')
    share3_img = Image.fromarray(share3, 'RGB')

    return share1_img, share2_img, share3_img

from PIL import Image
import numpy as np

def generate_black_and_white_image(intermediate_output_img):
    # Convert the PIL image to a numpy array
    intermediate_output_arr = np.array(intermediate_output_img)

    # Get the dimensions of the image
    height, width, _ = intermediate_output_arr.shape

    # Initialize a new array for the black and white image
    bw_image_arr = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel/module in the intermediate output image
    for i in range(height):
        for j in range(width):
            # Get the RGB values for the current pixel
            red_value = intermediate_output_arr[i, j, 0]
            green_value = intermediate_output_arr[i, j, 1]
            blue_value = intermediate_output_arr[i, j, 2]

            # Calculate the grayscale value using luminosity method
            gray_value = int(0.2989 * red_value + 0.5870 * green_value + 0.1140 * blue_value)

            # Set the grayscale value for the corresponding pixel in the black and white image array
            bw_image_arr[i, j] = gray_value

    # Create a PIL image from the black and white image array
    bw_image = Image.fromarray(bw_image_arr, 'L')  # 'L' mode for grayscale

    return bw_image



def generate_intermediate_output(share1, share2, share3):
    # Convert the PIL images to numpy arrays
    arr_share1 = np.array(share1)
    arr_share2 = np.array(share2)
    arr_share3 = np.array(share3)

    # Get the dimensions of the images
    height, width, _ = arr_share1.shape

    # Initialize the intermediate output image array
    intermediate_output_arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate through each pixel/module in the images
    for i in range(height):
        for j in range(width):
            # Get the RGB values for the corresponding pixels in each share
            color_share1 = arr_share1[i, j]
            color_share2 = arr_share2[i, j]
            color_share3 = arr_share3[i, j]

            red_value = color_share1[0] + color_share2[0] + color_share3[0]
            green_value = color_share1[1] + color_share2[1] + color_share3[1]
            blue_value = color_share1[2] + color_share2[2] + color_share3[2]

            if not (red_value > 200 and green_value > 200 and blue_value > 200):
                red_value = 255
                green_value = 0
                blue_value = 255

            # Calculate intermediate color components by summing
            # red_value = np.clip(color_share1[0] + color_share2[0] + color_share3[0], 0, 255)
            # green_value = np.clip(color_share1[1] + color_share2[1] + color_share3[1], 0, 255)
            # blue_value = np.clip(color_share1[2] + color_share2[2] + color_share3[2], 0, 255)
            #print(red_value, green_value, blue_value)
            #total = red_value + green_value + blue_value
            # if total < 750:
            #     intermediate_color = (255, green_value, 255)
            # else:


            intermediate_color = (red_value, green_value, blue_value)
                #print("Secondary total")
            # Create the intermediate color tuple


            # Assign the resulting color to the corresponding pixel in the intermediate output array
            intermediate_output_arr[i, j] = intermediate_color

    # Create a PIL image from the intermediate output array
    intermediate_output_img = Image.fromarray(intermediate_output_arr, 'RGB')

    return intermediate_output_img



def create_color_shares(qr_code_image_path):
    # Open the QR code image
    qr_code_image = Image.open(qr_code_image_path)

    # Convert the input image to grayscale
    gray_image = qr_code_image.convert("L")
    arr = np.array(gray_image)

    # Get the dimensions of the grayscale image
    height, width = arr.shape

    # Initialize color share images as empty arrays
    share1 = np.zeros((height, width, 3), dtype=np.uint8)
    share2 = np.zeros((height, width, 3), dtype=np.uint8)
    share3 = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate through each pixel/module in the grayscale image
    for i in range(height):
        for j in range(width):
            grayscale_value = arr[i, j]

            # Check if the pixel color in the original image is magenta
            original_pixel_color = qr_code_image.getpixel((j, i))
            is_magenta = (original_pixel_color == (255, 0, 255))
            random_number = random.randint(0, 2)
            # Determine unique colors for each share based on grayscale value
            if grayscale_value == 255:
                # White pixel (full brightness)
                if random_number == 0:
                    share1[i, j] = [255, 0, 0]  # Red in share1
                    share2[i, j] = [0, 255, 0]  # Green in share2
                    share3[i, j] = [0, 0, 255]  # Blue in share3
                if random_number == 1:
                    share1[i, j] = [0, 255, 0]  # Red in share1
                    share2[i, j] = [0, 0, 255]  # Green in share2
                    share3[i, j] = [255, 0, 0]  # Blue in share3
                if random_number == 2:
                    share1[i, j] = [0, 0, 255]  # Red in share1
                    share2[i, j] = [255, 0, 0]  # Green in share2
                    share3[i, j] = [0, 255, 0]  # Blue in share3
            elif is_magenta:
                # Magenta pixel (secondary color)
                if random_number == 0:
                    share1[i, j] = [255, 0, 0]  # Red in share1
                    share2[i, j] = [255, 0, 0]  # Red (same as share1) in share2
                    share3[i, j] = [0, 255, 0]  # Green in share3
                if random_number == 1:
                    share1[i, j] = [0, 0, 255]  # Red in share1
                    share2[i, j] = [0, 0, 255]  # Red (same as share1) in share2
                    share3[i, j] = [255, 0, 0]  # Green in share3
                if random_number == 2:
                    share1[i, j] = [0, 255, 0]  # Red in share1
                    share2[i, j] = [0, 255, 0]  # Red (same ased
                    # share1) in share2
                    share3[i, j] = [0, 0, 255]  # Green in share3

    # Create PIL images from the color shares
    share1_img = Image.fromarray(share1, 'RGB')
    share2_img = Image.fromarray(share2, 'RGB')
    share3_img = Image.fromarray(share3, 'RGB')

    return share1_img, share2_img, share3_img

def replace_black_color(qr_code_image_path, secondary_color=(255, 0, 255)):
    # Open the QR code image
    qr_code_image = Image.open(qr_code_image_path)

    # Convert the image to grayscale
    gray_image = qr_code_image.convert("L")

    # Convert the grayscale image to a NumPy array
    arr = np.array(gray_image)

    # Create a copy of the grayscale image array to work with color images
    colored_arr = np.stack([arr] * 3, axis=-1)

    # Define the RGB values for black and the secondary color (e.g., magenta)
    black_color = [0, 0, 0]
    secondary_color_rgb = np.array(secondary_color)

    # Replace black pixels with the secondary color in the color array
    colored_arr[np.where(arr == 0)] = secondary_color_rgb

    # Create a PIL image from the colored array
    intermediate_qr_code_image = Image.fromarray(colored_arr, 'RGB')

    return intermediate_qr_code_image


def overlay_shares(share1, share2, share3):
    # Proper overlay of three images requires a different approach:
    overlay = Image.blend(Image.blend(share1, share2, alpha=0.5), share3, alpha=0.5)
    return overlay


def get_qr_code_module_colors(qr_code_image_path):
    # Open the QR code image
    qr_code_image = Image.open(qr_code_image_path)

    # Get the dimensions of the image
    width, height = qr_code_image.size

    # Initialize a list to store the colors of each module
    module_colors = []

    # Iterate through each pixel/module in the image
    for y in range(height):
        row_colors = []  # Store colors for each row
        for x in range(width):
            # Get the RGB color tuple of the current pixel
            pixel_color = qr_code_image.getpixel((x, y))
            #print(pixel_color)
            # Append the color to the row_colors list
            row_colors.append(pixel_color)

        # Append the row_colors to the module_colors list
        module_colors.append(row_colors)
        # Print the dimensions of module_colors
    print(f"Dimensions of module_colors: {len(module_colors)} x {len(module_colors[0])}")
    return module_colors

def create_colored_pattern_image(module_colors):
    # Get the dimensions of module_colors
    height = len(module_colors)
    width = len(module_colors[0])

    # Create a new image with the same dimensions as module_colors
    pattern_image = Image.new("RGB", (width, height))

    # Iterate through each pixel/module in the image
    for y in range(height):
        for x in range(width):
            # Get the color of the current module
            color = module_colors[y][x]
            # Paste the color onto the pattern image at the corresponding position
            pattern_image.putpixel((x, y), color)

    return pattern_image

def process_qr_code_image(qr_code_image_path):
    # Open the QR code image
    qr_code_image = Image.open(qr_code_image_path)

    # Get the dimensions of the image
    width, height = qr_code_image.size

    # Initialize a list to store the colors of each module
    module_colors = []

    # Iterate through each pixel/module in the image
    for y in range(height):
        row_colors = []  # Store colors for each row
        for x in range(width):
            # Get the RGB color tuple of the current pixel
            pixel_color = qr_code_image.getpixel((x, y))

            # Append the color to the row_colors list
            row_colors.append(pixel_color)

        # Append the row_colors to the module_colors list
        module_colors.append(row_colors)

    # Create color shares based on the QR code image's module colors
    share1_img, share2_img, share3_img = create_color_shares(qr_code_image)

    return share1_img, share2_img, share3_img

from PIL import Image
import numpy as np

def generate_black_and_white_image(intermediate_output_img):
    # Convert the PIL image to a numpy array
    intermediate_output_arr = np.array(intermediate_output_img)

    # Get the dimensions of the image
    height, width, _ = intermediate_output_arr.shape

    # Initialize a new array for the black and white image
    bw_image_arr = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel/module in the intermediate output image
    for i in range(height):
        for j in range(width):
            # Get the RGB values for the current pixel
            red_value = intermediate_output_arr[i, j, 0]
            green_value = intermediate_output_arr[i, j, 1]
            blue_value = intermediate_output_arr[i, j, 2]

            # Calculate the grayscale value using luminosity method
            gray_value = int(0.2989 * red_value + 0.5870 * green_value + 0.1140 * blue_value)

            # Set the grayscale value for the corresponding pixel in the black and white image array
            bw_image_arr[i, j] = gray_value

    # Create a PIL image from the black and white image array
    bw_image = Image.fromarray(bw_image_arr, 'L')  # 'L' mode for grayscale

    return bw_image

def binarize_image(intermediate_output_img, threshold=128):
    # Convert the PIL image to a numpy array
    intermediate_output_arr = np.array(intermediate_output_img)

    # Get the dimensions of the image
    height, width = intermediate_output_arr.shape

    # Initialize a new array for the binarized image
    binarized_image_arr = np.zeros((height, width), dtype=np.uint8)

    # Iterate through each pixel/module in the intermediate output image
    for i in range(height):
        for j in range(width):
            # Get the grayscale value for the current pixel
            gray_value = intermediate_output_arr[i, j]

            # Apply thresholding to binarize the image
            if gray_value >= threshold:
                binarized_image_arr[i, j] = 255  # Set as white (255)
            else:
                binarized_image_arr[i, j] = 0    # Set as black (0)

    # Create a PIL image from the binarized image array
    binarized_image = Image.fromarray(binarized_image_arr, 'L')  # 'L' mode for grayscale

    return binarized_image



def main():
    #Step 0 - Preparing input data
    #   Generate a sample secret QR code (replace 'secret_data' with your data)
    #   Make sure the secret_data is a multiple of ec_codewords_per_block for the selected version, w.r.t num of words
    secret_data = "I'm Nasser, telecom student at Shandong University of Science and Technology. I am happy.."
    input_image_path = "images/input/original_qr_code.png";
    intermediate_image_path = "images/output/intermediate_qr_code.png"
    share1_output_path = "images/output/share1.png"
    share2_output_path = "images/output/share2.png"
    share3_output_path = "images/output/share3.png"
    gray_image_path = "images/output/grayscale_image.png"
    binarized_image_path = "images/output/binarized_image.png"
    print(len(secret_data))
    qr_code_size = 30
    qr_code_version = 3
    qr_code = generate_secret_qr_code(secret_data, qr_code_version, qr_code_size)
    qr_code.save(input_image_path, "PNG")
    qr_code.show()
    #Reading a given QR_code as input
    qr_code_image = Image.open(input_image_path)

    module_colors = get_qr_code_module_colors(input_image_path)
    secondary_color = (255, 0, 255)  # Magenta color
    intermediate_qr_code = replace_black_color(input_image_path, secondary_color)
    intermediate_qr_code.save(intermediate_image_path, "PNG")
    pattern_image = create_colored_pattern_image(module_colors)
    pattern_image.show()  # Display the pattern image
    share1, share2, share3 = create_color_shares(intermediate_image_path)
    share1.save(share1_output_path, "PNG")
    share2.save(share2_output_path, "PNG")
    share3.save(share3_output_path, "PNG")
    # Assuming you have generated the three share images (share1, share2, share3)
    # Call the function to generate the intermediate output image
    intermediate_output_img = generate_intermediate_output(share1, share2, share3)

    # Save or display the intermediate output image
    intermediate_output_img.save(intermediate_image_path, "PNG")
    intermediate_output_img.show()
    grayscale_image = generate_black_and_white_image(intermediate_output_img)
    grayscale_image.save(gray_image_path, "PNG")
    # Binarize the intermediate output image with a threshold of 128
    binary_image = binarize_image(grayscale_image, threshold=128)
    binary_image.save(binarized_image_path, "PNG")
if __name__ == "__main__":
    main()
