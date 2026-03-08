import os


def load_20newsgroups_dataset(data_dir="data/20_newsgroups"):
    documents = []
    labels = []
    label_names = []

    # Each folder represents a topic
    for label_id, category in enumerate(os.listdir(data_dir)):

        category_path = os.path.join(data_dir, category)

        if not os.path.isdir(category_path):
            continue

        label_names.append(category)

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            try:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read()
                    documents.append(text)
                    labels.append(label_id)

            except:
                # Skip files with encoding issues
                continue

    return documents, labels, label_names