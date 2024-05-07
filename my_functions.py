from PIL import Image
import numpy as np
import qrcode
import random
import cv2
from pyzbar.pyzbar import decode
import galois
import matplotlib.pyplot as plt
def generate_secret_qr_code(secret_data, version=1, size=10):
    qr = qrcode.QRCode(version=version, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=size, border=4)
    qr.add_data(secret_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    return qr_img

def extract_data_encoding_type(qr_code_image):
    # Decode the QR code image
    decoded_data = decode(qr_code_image)

    if decoded_data:
        # Extract the data encoding type from the decoded QR code
        data_encoding_type = decoded_data[0].type

        # Return the data encoding type
        return data_encoding_type
    else:
        # Return None if QR code decoding failed
        return None

def extract_bit_stream(qr_code_image):
    # Decode the QR code image
    decoded_data = decode(qr_code_image)

    if decoded_data:
        # Extract the binary data (bit stream) from the QR code
        bit_stream = decoded_data[0].data

        # Convert bit stream bytes to binary string
        bit_stream_binary = ''.join(format(byte, '08b') for byte in bit_stream)

        return bit_stream_binary
    else:
        return None

def get_qr_code_version(bit_stream):
    # Define the mapping of QR code versions to bit stream length ranges
    version_ranges = {
        1: (208, 272),   # QR code version 1 (21x21 modules)
        2: (368, 440),   # QR code version 2 (25x25 modules)
        3: (648, 720),   # QR code version 3 (29x29 modules)
        # Add more versions and their bit stream length ranges as needed
    }
    bit_stream_length = len(bit_stream)
    print("Bit stream length", bit_stream_length)
    # Iterate over each version and its corresponding valid bit stream length range
    for version, (min_length, max_length) in version_ranges.items():
        #Check if the detected bit stream length falls within the range for the current version
        if min_length <= bit_stream_length <= max_length:
            print(f"Detected QR code version: {version}")
            return version

    # Return None if no matching version is found
    print("QR code version detection failed.")
    return None


def convert_to_codewords(bit_stream_binary, version):
    # Calculate maximum number of codewords based on QR code version
    total_codewords = QR_CODE_VERSION_INFO[version]['total_codewords']

    # Split bit stream into codewords
    codewords = [bit_stream_binary[i:i+8] for i in range(0, len(bit_stream_binary), 8)]

    # Pad with zeros if necessary to reach total number of codewords
    if len(codewords) < total_codewords:
        codewords += ['00000000'] * (total_codewords - len(codewords))

    return codewords

def assemble_blocks(codewords, ec_codewords_per_block):
    # Calculate the number of data blocks
    print("Numb of code words: ", len(codewords))
    num_blocks = len(codewords) // ec_codewords_per_block

    # Check if the length of codewords is a multiple of ec_codewords_per_block
    if len(codewords) % ec_codewords_per_block != 0:
        raise ValueError("Number of codewords is not a multiple of ec_codewords_per_block")

    # Reshape codewords into data blocks with error correction codewords
    data_blocks = np.array(codewords).reshape(num_blocks, ec_codewords_per_block)

    return data_blocks

# QR Code version information (total codewords and error correction codewords per block)
QR_CODE_VERSION_INFO = {
    1: {'total_codewords': 26, 'ec_codewords_per_block': 7},
    2: {'total_codewords': 44, 'ec_codewords_per_block': 10},
    3: {'total_codewords': 70, 'ec_codewords_per_block': 15},
    # Add more versions as needed...
}

def get_data_blocks(bit_stream, qr_code_version):
    if bit_stream:
        # Determine QR code version (e.g., based on bit stream length)
        version = get_qr_code_version(bit_stream)
        if version:
            print(f"Detected QR code version: {version}")


            # Calculate the target length (a multiple of ec_codewords_per_block)
            ec_codewords_per_block = QR_CODE_VERSION_INFO[qr_code_version]['ec_codewords_per_block']
            target_length = ((
                                         len(bit_stream) + ec_codewords_per_block - 1) // ec_codewords_per_block) * ec_codewords_per_block
            print("Target length : ", target_length)
            # Adjust the bit stream length to match the target length
            adjusted_bit_stream = bit_stream.ljust(target_length, '0')[:target_length]

            # Convert bit stream to codewords
            codewords = convert_to_codewords(adjusted_bit_stream, qr_code_version)


            # Convert bit stream to codewords
            codewords = convert_to_codewords(bit_stream, version)
            if codewords:
                # Assemble codewords into data blocks with error correction
                ec_codewords_per_block = QR_CODE_VERSION_INFO[version]['ec_codewords_per_block']
                data_blocks = assemble_blocks(codewords, ec_codewords_per_block)
                # Print or use the assembled data blocks for ECC encoding
                print("Data Blocks:")
                print(data_blocks)
                return data_blocks
            else:
                print("Failed to convert bit stream to codewords.")
        else:
            print("QR code version detection failed.")
    else:
        print("QR code decoding failed or no data found.")


def perform_error_correction(data_blocks, qr_code_version):
    # Determine the number of error correction codewords per block based on QR code version
    ec_codewords_per_block = QR_CODE_VERSION_INFO[qr_code_version]['ec_codewords_per_block']

    encoded_data_blocks = []
    rs = galois.ReedSolomon(255, 253);

    for data_block in data_blocks:
        # Calculate the number of data codewords in this block
        data_codewords = len(data_block)

        # Calculate the number of total codewords (data + error correction)
        total_codewords = data_codewords + ec_codewords_per_block

        # Perform Reed-Solomon encoding for error correction
        # Use a Reed-Solomon encoding library or implementation
        # Example: using the `rs_encode` function from the `pyqrcode` library
        error_correction_codewords = rs.encode(data_block, ec_codewords_per_block)

        # Append error correction codewords to the data block
        encoded_data_block = data_block + error_correction_codewords
        encoded_data_blocks.append(encoded_data_block)

    return encoded_data_blocks

def perform_reed_solomon_encoding(data_blocks, ec_codewords_per_block):
    # Initialize Galois field GF(2^8) for Reed-Solomon encoding
    GF = galois.GF(2**8)

    # Determine the codeword size (n) and message size (k) based on the data
    #n = len(data_blocks[0]) + ec_codewords_per_block  # Total codeword size
    #k = len(data_blocks[0])  # Message size
    n = 255  # Total codeword size for a standard QR code (version 3)
    k = 223  # Message size (data capacity) for a standard QR code (version 3)

    # Create a Reed-Solomon code instance with the specified parameters
    rs = galois.ReedSolomon(n, k)

    # Encode each data block
    encoded_blocks = []
    for block in data_blocks:
        # Convert binary string to integer list
        data = [int(bit, 2) for bit in block]
        # Encode using Reed-Solomon
        encoded = rs.encode(data)
        # Append the encoded data block
        encoded_blocks.append(encoded)

    return encoded_blocks



def encode_vss_qr(secret_qr_img):
    # Convert the secret QR code image to a NumPy array
    secret_qr_array = np.array(secret_qr_img)

    # Define the sets of colors for encoding black and white secret modules
    E0 = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Primary colors (red, green, blue)
    E1 = [(255, 255, 255)]  # White color

    # Initialize the shares
    shares = []
    for _ in range(3):
        shares.append(Image.new('RGB', secret_qr_img.size))

    # Iterate through each module of the secret QR code
    for x in range(secret_qr_img.size[0]):
        for y in range(secret_qr_img.size[1]):
            # Check the color of the current module (black or white)
            if np.all(secret_qr_array[y][x] == 0):  # Black module
                colors = E0
            else:  # White module
                colors = E1

            # Shuffle the colors list and select the first three colors for each share
            random.shuffle(colors)
            selected_colors = colors[:3]

            # Assign colors to each share
            for i, color in enumerate(selected_colors):
                shares[i].putpixel((x, y), color)

    return shares


def stack_shares(shares):
    """
    Stack the three shares pixel by pixel to reconstruct the color QR code T.
    """
    width, height = shares[0].size
    stacked_img = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            pixel_values = [share.getpixel((x, y)) for share in shares]
            stacked_img.putpixel((x, y), tuple(np.sum(pixel_values, axis=0) // 3))

    return stacked_img


def color_to_grayscale(img):
    """
    Convert the color QR code image to grayscale using weighted average method.
    """
    weighted_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    weighted_img = cv2.cvtColor(weighted_img, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(weighted_img)


def binarize_image(img):
    """
    Binarize the grayscale QR code image using Otsu's algorithm.
    """
    _, binarized_img = cv2.threshold(np.array(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binarized_img)

def place_codewords_in_qr_matrix(qr_code_matrix, encoded_blocks, masking_pattern):
    # Determine dimensions and positions for placing codewords in the QR code matrix
    qr_size = len(qr_code_matrix)
    module_positions = get_module_positions(qr_size)  # Determine positions for data modules

    # Insert codewords into the QR code matrix
    for block_index, block in enumerate(encoded_blocks):
        for index, value in enumerate(block):
            x, y = module_positions[block_index][index]  # Get position for current codeword
            qr_code_matrix[x][y] = value

    # Apply masking pattern to the QR code matrix
    apply_masking(qr_code_matrix, masking_pattern)

    # Add format and version information to the QR code matrix
    add_format_and_version(qr_code_matrix, masking_pattern)

    return qr_code_matrix

def get_module_positions(qr_size):
    # Determine positions for placing codewords in the QR code matrix based on QR size
    # Implement logic to map block indices to module positions
    # Example: Generate module positions for a specific QR size and version
    module_positions = [[(x, y) for y in range(qr_size)] for x in range(qr_size)]
    return module_positions

def apply_masking(qr_code_matrix, masking_pattern):
    # Implement masking logic based on the chosen masking pattern
    # Apply masking to the QR code matrix
    pass

def add_format_and_version(qr_code_matrix, masking_pattern):
    # Implement adding format and version information to the QR code matrix
    # Include logic to calculate and embed format and version information
    pass

def initialize_qr_code_matrix(qr_size):
    # Create a blank QR code matrix as a PIL image
    qr_code_matrix = Image.new("1", (qr_size, qr_size), 1)  # Mode "1" for black and white, size (qr_size, qr_size)
    return qr_code_matrix

def choose_masking_pattern(qr_code_matrix):
    min_penalty = float('inf')  # Initialize with a large value
    best_pattern = 0

    for pattern in range(8):  # Iterate through masking patterns 0 to 7
        masked_matrix = apply_masking_pattern(qr_code_matrix, pattern)
        penalty = calculate_penalty(masked_matrix)

        if penalty < min_penalty:
            min_penalty = penalty
            best_pattern = pattern

    return best_pattern

def apply_masking_pattern(qr_code_matrix, pattern):
    # Apply the specified masking pattern to the QR code matrix
    # Modify the matrix according to the rules of the masking pattern
    # Return the masked matrix
    masked_matrix = qr_code_matrix  # Placeholder implementation

    return masked_matrix

def calculate_penalty(qr_code_matrix):
    # Calculate penalty score for the given masked QR code matrix
    # Penalize based on patterns of identical modules and other criteria
    penalty = 0  # Placeholder implementation

    return penalty

def display_qr_code(qr_code_matrix):
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Display the QR code matrix as an image
    ax.imshow(qr_code_matrix, cmap='binary', interpolation='nearest')

    # Hide the axes
    ax.axis('off')

    # Set the title
    ax.set_title('QR Code')

    # Show the plot
    plt.show()

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