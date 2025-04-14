import os
import xml.etree.ElementTree as ET
from PIL import Image

# Paths
image_dir = r"C:\Users\raksh\Downloads\helmet_data\images"
annotation_dir = r"C:\Users\raksh\Downloads\helmet_data\annotations"
output_label_dir = r"C:\Users\raksh\Downloads\helmet_data\labels"

# Make sure label output dir exists
os.makedirs(output_label_dir, exist_ok=True)

# Optional: Map class names to IDs
classes = ["With Helmet", "Without helmet"]  # <- update your actual class names

def convert_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return [x_center * dw, y_center * dh, w * dw, h * dh]

for xml_file in os.listdir(annotation_dir):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(annotation_dir, xml_file))
    root = tree.getroot()

    image_name = root.find('filename').text
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Image not found for {xml_file}, skipping.")
        continue

    img = Image.open(image_path)
    w, h = img.size

    yolo_output = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymin = int(float(bbox.find('ymin').text))
        ymax = int(float(bbox.find('ymax').text))

        yolo_box = convert_to_yolo((w, h), (xmin, xmax, ymin, ymax))
        yolo_output.append(f"{cls_id} {' '.join(f'{a:.6f}' for a in yolo_box)}")

    # Save to .txt
    txt_file = os.path.splitext(xml_file)[0] + ".txt"
    with open(os.path.join(output_label_dir, txt_file), "w") as f:
        f.write("\n".join(yolo_output))
