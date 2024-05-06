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

def main():
    # Generate a secret QR code (replace 'secret_data' with your data)
    secret_data = "I'm Nasser, telecom student at Shandong University of Science and Technology."
    #Get the secret QR code as input
    secret_qr_img = generate_secret_qr_code(secret_data)

    # Encode the secret QR code using VSS-QR algorithm
    shares = encode_vss_qr(secret_qr_img)

    # Save the shares as images
    for i, share in enumerate(shares):
        share.save(f"share_{i + 1}.png")

        # Load the three shares
        shares = [Image.open(f"share_{i}.png") for i in range(1, 4)]

        # Stack the shares to reconstruct the color QR code T
        stacked_img = stack_shares(shares)

        # Convert the color QR code to grayscale
        grayscale_img = color_to_grayscale(stacked_img)

        # Binarize the grayscale QR code image
        binarized_img = binarize_image(grayscale_img)

        # Save the binarized QR code image
        binarized_img.save("binarized_qr_code.png")


if __name__ == "__main__":
    main()
