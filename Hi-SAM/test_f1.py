from utils.f1_score_for_bb import compute_f1 
import cv2
import xml.etree.ElementTree as ET
import time


page_id = "Seite0407"
img_path = f"../../DATASETS/READ_2016/Test/Images/{page_id}.JPG"
gt_path = f"../../DATASETS/READ_2016/Test/gt/{page_id}.xml"
pred_path = f"../../DATASETS/READ_2016/Test/READ_2016_H060/{page_id}.txt"

tree = ET.parse(gt_path)
root = tree.getroot()
ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

i=1
image = cv2.imread(img_path)
gt_xyxy, pred_xyxy = [], []
for line in root.findall(".//pc:TextLine", ns):
    coords_elem = line.find("pc:Coords", ns)
    points_str = coords_elem.attrib['points']
    points = [tuple(map(int, p.split(','))) for p in points_str.split()]
    # Compute bounding rectangle from polygon
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    points = [x1, y1, x2, y2]
    gt_xyxy.append(points)
    i+=1

with open(pred_path, "r") as file:
    for line in file:
        line = line.strip()
        box = line.split(" ")
        box = list(map(int, box))
        x1, y1, x2, y2 = map(int, box)
        points = [x1, y1, x2, y2]
        pred_xyxy.append(points)  

start_time = time.time()
scores = compute_f1(gt_xyxy, pred_xyxy)
print(scores)

print(f"Runtime: {time.time() - start_time:.4f} seconds")