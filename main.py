from my_functions import (
    generate_secret_qr_code,
    get_qr_code_module_colors,
    replace_black_color,
    create_colored_pattern_image,
    create_color_shares,
    generate_intermediate_output,
    generate_black_and_white_image,
    binarize_image)
from PIL import Image

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
