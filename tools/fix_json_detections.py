import json
import os

json_path = os.path.join('/home/vitor/python/Detectron.pytorch/tools/detections_100.json')
json_save_path = os.path.join('/home/vitor/python/Detectron.pytorch/tools/detections_100_fixed.json')
detectections_json = json.load(open(json_path, 'r'))

for detection in detectections_json:
    detection['image_id'] = int('2018' + str(detection['image_id']).zfill(7))

with open(json_save_path, 'w') as fp:
    json.dump(detectections_json, fp)

