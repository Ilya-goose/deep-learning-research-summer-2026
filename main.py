import os


def read_protocol(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                key = parts[1]
                value = parts[4]
                data[key] = value
    return data


if __name__ == "__main__":
    filename = "ASVspoof2019.LA.cm.eval.trl.txt"
    if os.path.exists(filename):
        labels = read_protocol(filename)
        print(len(labels))
        print(list(labels.items())[:5])
    else:
        print("File not found")