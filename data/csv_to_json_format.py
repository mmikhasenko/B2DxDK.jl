import csv
import json

def convert_csv_to_json(data_filename, error_filename):
    with open(data_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        waves = header[1:]

    data_matrix = []
    with open(data_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            data_matrix.append([float(x) for x in row[1:] if x])

    error_matrix = []
    with open(error_filename, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            error_matrix.append([float(x) for x in row[1:] if x])

    # Manuelles Erstellen der formatierten JSON-Strings
    waves_str = json.dumps(waves, indent=2)
    matrix_str = "[\n" + ",\n".join([f"  {json.dumps(row)}" for row in data_matrix]) + "\n]"
    error_str = "[\n" + ",\n".join([f"  {json.dumps(row)}" for row in error_matrix]) + "\n]"

    with open('interference_python.json', 'w') as f:
        f.write("{\n")
        f.write(f'  "waves": {waves_str.strip()},\n')
        f.write(f'  "matrix": {matrix_str},\n')
        f.write(f'  "uncertainty": {error_str}\n')
        f.write("}\n")


data_file = 'fit_frac1_pw.csv'
error_file = 'fit_frac1_pw_err.csv'

convert_csv_to_json(data_file, error_file)

print("The JSON-file 'interference_python.json' has been created.")