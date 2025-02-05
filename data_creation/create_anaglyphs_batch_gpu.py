import torch
import cv2 as cv
import os
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split

SOURCE_DIR = '/data/raw'
TARGET_DIR = '/data/'
CHUNK_WIDTH = 256
CHUNK_HEIGHT = 256
CSV_FILE = os.path.join(TARGET_DIR, 'image_locations.csv')
TRAIN_CSV_FILE = os.path.join(TARGET_DIR, 'train.csv')
VALIDATION_CSV_FILE = os.path.join(TARGET_DIR, 'validation.csv')
TEST_CSV_FILE = os.path.join(TARGET_DIR, 'test.csv')
TRAIN_TEST_RATIO = 0.7
TRAIN_VAL_RATIO = 0.8

# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_image(image):
    """Split the image into left and right halves"""
    height, width, _ = image.shape
    middle = width // 2
    left_img = image[:, :middle]
    right_img = image[:, middle:]
    return left_img, right_img

def scale_image(img):
    """Scale image to 2560 x 720"""
    if img.shape[0] < 720 or img.shape[1] < 2560:
        img = cv.resize(img, (2560, 720))
    return img

def create_image_chunks(image):
    """Split the image into chunks of size CHUNK_WIDTH x CHUNK_HEIGHT"""
    img_height, img_width = image.shape[:2]
    num_rows = (img_height // CHUNK_HEIGHT) + 1
    num_cols = (img_width // CHUNK_WIDTH) + 1
    overlap_y = (img_height - num_rows * CHUNK_HEIGHT) // num_rows
    overlap_x = (img_width - num_cols * CHUNK_WIDTH) // num_cols
    chunks = []
    for row in range(num_rows):
        for col in range(num_cols):
            y_start = row * (CHUNK_HEIGHT - overlap_y)
            x_start = col * (CHUNK_WIDTH - overlap_x)
            y_end = min(y_start + CHUNK_HEIGHT, img_height)
            x_end = min(x_start + CHUNK_WIDTH, img_width)
            if y_end - y_start == CHUNK_HEIGHT and x_end - x_start == CHUNK_WIDTH:
                chunks.append(image[y_start:y_end, x_start:x_end])
    return chunks

def merge_channels(img_left, img_right):
    """Merge the channels of the left and right images"""
    b_left, g_left, r_left = cv.split(img_left)
    b_right, g_right, r_right = cv.split(img_right)
    merged_img = cv.merge((b_right, g_right, r_left))
    merged_img_reversed = cv.merge((b_left, g_left, r_right))
    return merged_img, merged_img_reversed

def process_image(img_input_raw):
    """Process single image using CUDA"""
    img_scaled_full = scale_image(img_input_raw)
    img_left, img_right = split_image(img_scaled_full)
    chunks_left_arr = create_image_chunks(img_left)
    chunks_right_arr = create_image_chunks(img_right)

    anaglyphs_arr = []
    anaglyphs_reversed_arr = []
    for chunk in zip(chunks_left_arr, chunks_right_arr):
        left_chunk = torch.tensor(chunk[0]).to(device)
        right_chunk = torch.tensor(chunk[1]).to(device)
        anaglyph, anaglyph_reversed = merge_channels(left_chunk.cpu().numpy(), right_chunk.cpu().numpy())
        anaglyphs_arr.append(anaglyph)
        anaglyphs_reversed_arr.append(anaglyph_reversed)

    return img_scaled_full, img_left, img_right, chunks_left_arr, chunks_right_arr, anaglyphs_arr, anaglyphs_reversed_arr

def save_image_locations_to_csv(chunks_left_arr, chunks_right_arr, anaglyphs_arr, anaglyphs_reversed_arr, image_name,
                                target_path, csv_file):
    """Save the locations of all parts of the processed image into a CSV file."""
    base_name, ext = os.path.splitext(image_name)
    locations = [
        (image_name, f"{base_name}_scaled{ext}", os.path.join(target_path, 'scaled', f"{base_name}_scaled{ext}"), 'scaled'),
        (image_name, f"{base_name}_left{ext}", os.path.join(target_path, 'left', f"{base_name}_left{ext}"), 'left'),
        (image_name, f"{base_name}_right{ext}", os.path.join(target_path, 'right', f"{base_name}_right{ext}"), 'right'),
    ]
    for idx in range(len(chunks_left_arr)):
        locations.append((image_name, f"{base_name}_left_chunk_{idx}{ext}",
                          os.path.join(target_path, 'left_chunks', f"{base_name}_left_chunk_{idx}{ext}"), 'left_chunk'))
    for idx in range(len(chunks_right_arr)):
        locations.append((image_name, f"{base_name}_right_chunk_{idx}{ext}",
                          os.path.join(target_path, 'right_chunks', f"{base_name}_right_chunk_{idx}{ext}"), 'right_chunk'))
    for idx in range(len(anaglyphs_arr)):
        locations.append((image_name, f"{base_name}_anaglyph_{idx}{ext}",
                          os.path.join(target_path, 'anaglyphs', f"{base_name}_anaglyph_{idx}{ext}"), 'anaglyph'))
    for idx in range(len(anaglyphs_reversed_arr)):
        locations.append((image_name, f"{base_name}_anaglyph_reversed_{idx}{ext}",
                          os.path.join(target_path, 'anaglyphs_reversed', f"{base_name}_anaglyph_reversed_{idx}{ext}"), 'anaglyph_reversed'))
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Raw Image Name', 'Processed Image Name', 'Path', 'Label'])
        for location in locations:
            writer.writerow(location)

def split_csv_file(csv_file, train_csv_file, test_csv_file, split_ratio):
    """Split the data in the CSV file into training and testing sets based on the first column."""
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)
    grouped_rows = {}
    for row in rows:
        key = row[0]
        if key not in grouped_rows:
            grouped_rows[key] = []
        grouped_rows[key].append(row)
    keys = list(grouped_rows.keys())
    data = [grouped_rows[key] for key in keys]
    train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    train_rows = [item for sublist in train_data for item in sublist]
    test_rows = [item for sublist in test_data for item in sublist]
    with open(train_csv_file, 'w', newline='') as train_file, open(test_csv_file, 'w', newline='') as test_file:
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)
        train_writer.writerow(header)
        test_writer.writerow(header)
        train_writer.writerows(train_rows)
        test_writer.writerows(test_rows)

def ensure_directories_exist(directories):
    """Check if all directories are present and create them if not."""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def process_all_images(source_path, target_path, pbar):
    files = os.listdir(source_path)
    images = [file for file in files if file.endswith('.jpg')]
    for image in images:
        try:
            image_path = os.path.join(source_path, image)
            img = cv.imread(image_path)
            img_scaled_full, img_left, img_right, chunks_left_arr, chunks_right_arr, anaglyphs_arr, anaglyphs_reversed_arr = process_image(img)
            base_name, ext = os.path.splitext(image)
            scaled_img_path = os.path.join(target_path, 'scaled', base_name + '_scaled' + ext)
            cv.imwrite(scaled_img_path, img_scaled_full)
            left_img_path = os.path.join(target_path, 'left', base_name + '_left' + ext)
            cv.imwrite(left_img_path, img_left)
            left_img_path = os.path.join(target_path, 'right', base_name + '_right' + ext)
            cv.imwrite(left_img_path, img_right)
            for idx, left_chunk in enumerate(chunks_left_arr):
                chunk_path = os.path.join(target_path, 'left_chunks', base_name + f"_left_chunk_{idx}" + ext)
                cv.imwrite(chunk_path, left_chunk)
            for idx, right_chunk in enumerate(chunks_right_arr):
                chunk_path = os.path.join(target_path, 'right_chunks', base_name + f"_right_chunk_{idx}" + ext)
                cv.imwrite(chunk_path, right_chunk)
            for idx, anaglyph in enumerate(anaglyphs_arr):
                anaglyph_path = os.path.join(target_path, 'anaglyphs', base_name + f"_anaglyph_{idx}" + ext)
                cv.imwrite(anaglyph_path, anaglyph)
            for idx, anaglyph_reversed in enumerate(anaglyphs_reversed_arr):
                anaglyph_reversed_path = os.path.join(target_path, 'anaglyphs_reversed', base_name + f"_anaglyph_reversed_{idx}" + ext)
                cv.imwrite(anaglyph_reversed_path, anaglyph_reversed)
            save_image_locations_to_csv(chunks_left_arr, chunks_right_arr, anaglyphs_arr, anaglyphs_reversed_arr, image, target_path, CSV_FILE)
        except Exception as e:
            print(f"Error processing image '{image}': {e}")
        pbar.update()

if __name__ == '__main__':
    directories_to_check = [
        os.path.join(TARGET_DIR, 'scaled'),
        os.path.join(TARGET_DIR, 'left'),
        os.path.join(TARGET_DIR, 'right'),
        os.path.join(TARGET_DIR, 'left_chunks'),
        os.path.join(TARGET_DIR, 'right_chunks'),
        os.path.join(TARGET_DIR, 'anaglyphs'),
        os.path.join(TARGET_DIR, 'anaglyphs_reversed')
    ]
    ensure_directories_exist(directories_to_check)
    total_files = sum([len(files) for r, d, files in os.walk(SOURCE_DIR) if any(f.endswith('.jpg') for f in files)])
    with tqdm(total=total_files, desc="Processing images", unit="file") as pbar:
        process_all_images(SOURCE_DIR, TARGET_DIR, pbar)
    temp_train_csv_file = os.path.join(TARGET_DIR, 'train_and_val.csv')
    split_csv_file(CSV_FILE, temp_train_csv_file, TEST_CSV_FILE, TRAIN_TEST_RATIO)
    split_csv_file(temp_train_csv_file, TRAIN_CSV_FILE, VALIDATION_CSV_FILE, TRAIN_VAL_RATIO)
    print("CSV file split into training, validation and testing sets.")
    print("Processing complete.")