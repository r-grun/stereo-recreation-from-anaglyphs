import csv
import os

# Base directory
base_dir = '../../data'

# Paths and target labels
train_csv = os.path.join(base_dir, 'train.csv')
train_anaglyph_output = os.path.join(base_dir, 'train_anaglyphs.txt')
train_reversed_output = os.path.join(base_dir, 'train_reversed.txt')
train_left_output = os.path.join(base_dir, 'train_left.txt')
train_right_output = os.path.join(base_dir, 'train_right.txt')

test_csv = os.path.join(base_dir, 'test.csv')
test_anaglyph_output = os.path.join(base_dir, 'test_anaglyphs.txt')
test_reversed_output = os.path.join(base_dir, 'test_reversed.txt')
test_left_output = os.path.join(base_dir, 'test_left.txt')
test_right_output = os.path.join(base_dir, 'test_right.txt')

validation_csv = os.path.join(base_dir, 'validation.csv')
validation_anaglyph_output = os.path.join(base_dir, 'validation_anaglyphs.txt')
validation_reversed_output = os.path.join(base_dir, 'validation_reversed.txt')
validation_left_output = os.path.join(base_dir, 'validation_left.txt')
validation_right_output = os.path.join(base_dir, 'validation_right.txt')


def filter_csv(input_csv, output_txt, target_label):
    """
    Filters rows from a CSV file based on a target label and writes the 'Path' column values to a text file.

    :param input_csv: Path to the input CSV file.
    :param output_txt: Path to the output text file.
    :param target_label: The label to filter rows by.
    """
    try:
        with open(input_csv, mode='r', newline='', encoding='utf-8') as csv_file, open(output_txt, mode='w', encoding='utf-8') as txt_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                if row['Label'] == target_label:
                    txt_file.write(row['Path'] + '\n')

        print(f"Filtered rows with label '{target_label}' written to {output_txt}")
    except FileNotFoundError:
        print(f"Error: The file {input_csv} was not found.")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Filter rows and create the output files

# Train
filter_csv(train_csv, train_anaglyph_output, 'anaglyph')
# filter_csv(train_csv, train_reversed_output, 'anaglyph_reversed')
filter_csv(train_csv, train_left_output, 'left_chunk')
filter_csv(train_csv, train_right_output, 'right_chunk')

# Test
filter_csv(test_csv, test_anaglyph_output, 'anaglyph')
# filter_csv(test_csv, test_reversed_output, 'anaglyph_reversed')
filter_csv(test_csv, test_left_output, 'left_chunk')
filter_csv(test_csv, test_right_output, 'right_chunk')

# Validation
filter_csv(validation_csv, validation_anaglyph_output, 'anaglyph')
# filter_csv(validation_csv, validation_reversed_output, 'anaglyph_reversed')
filter_csv(validation_csv, validation_left_output, 'left_chunk')
filter_csv(validation_csv, validation_right_output, 'right_chunk')

