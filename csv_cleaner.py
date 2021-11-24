import csv
import sys

file_name = sys.argv[1]

data = []

with open(f'{file_name}.csv', 'r') as original:

    data_reader = csv.reader(original, delimiter=",")
    
    header_read = False

    for row in data_reader:

        if not header_read:
            data.append(row)
            header_read = True
            continue

        new_row = row[0].split(",")
        data.append(new_row)

    original.close()

with open(f'{file_name}_clean.csv', 'w') as clean_v:

    data_writer = csv.writer(clean_v)

    for row in data:

        data_writer.writerow(row)

    clean_v.close()

print("Dataset limpiado exitosamente")

        