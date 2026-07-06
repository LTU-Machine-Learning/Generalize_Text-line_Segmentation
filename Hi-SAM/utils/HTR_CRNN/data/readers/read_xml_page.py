import xml.etree.ElementTree as ET
import html

def strip_namespace(tree):
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]  # strip all namespaces

def read_xml_page_gt(path_file):
    tree = ET.parse(path_file)
    root = tree.getroot()
    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    ## Extract GT Points ######################
    gt_points, gt_texts = [], []
    for line in root.findall(".//pc:TextLine", ns):
        coords_elem = line.find("pc:Coords", ns)
        points_str = coords_elem.attrib['points']
        points = [tuple(map(int, p.split(','))) for p in points_str.split()]
        gt_points.append(points)

        line_unicode = line.findtext("pc:TextEquiv/pc:Unicode", default=None, namespaces=ns)
        gt_texts.append(line_unicode.strip())

    gt_boxes = []
    for poly in gt_points:
        gt_boxes.append(polygon_to_bbox(poly))
    return  gt_boxes, gt_texts

def polygon_to_bbox(poly):
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max, y_max]
