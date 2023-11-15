import glob
import json
import numpy as np
import torch
import torch.cuda
import clip
from PIL import Image
import cv2
from clip import tokenize
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

paths = glob.glob('./初赛测试视频/*')
paths.sort()

current_date = datetime.today().date().strftime("%Y%m%d")[2:]

submit_json = {
    "author": "jzxd",
    "time": current_date,
    "model": "VitB/32",
    "test_results": []
}

en_match_words = {
"scerario" : ["suburbs","city street","expressway","tunnel","parking-lot","gas or charging stations","unknown"],
"weather" : ["clear","cloudy","raining","foggy","snowy","unknown"],
"period" : ["daytime","dawn or dusk","night","unknown"],
"road_structure" : ["normal","crossroads","T-junction","ramp","lane merging","parking lot entrance","round about","unknown"],
"general_obstacle" : ["nothing","speed bumper","traffic cone","water horse","stone","manhole cover","nothing","unknown"],
"abnormal_condition" : ["uneven","oil or water stain","standing water","cracked","nothing","unknown"],
"ego_car_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
"closest_participants_type" : ["passenger car","bus","truck","pedestrain","policeman","nothing","others","unknown"],
"closest_participants_behavior" : ["slow down","go straight","turn right","turn left","stop","U-turn","speed up","lane change","others"],
}

for video_path in paths:
    print(video_path)
    clip_id = video_path.split('/')[-1]
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,total_frames/2)
    img = cap.read()[1]

    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0).to(device)

    single_video_result = {
        "clip_id": clip_id,
        "scerario": "city street",
        "weather": "clear",
        "period": "daytime",
        "road_structure": "normal",
        "general_obstacle": "nothing",
        "abnormal_condition": "nothing",
        "ego_car_behavior": "go straight",
        "closest_participants_type": "passenger car",
        "closest_participants_behavior": "slow down"
    }

    for keyword in en_match_words.keys():
        if keyword not in ["weather", "road_structure","period","scerario"]:
            continue
        texts = np.array(en_match_words[keyword])
        text = clip.tokenize(en_match_words[keyword]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(tokenize(en_match_words[keyword]).to(device))                      
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs = np.array(probs)
        if(keyword == 'scerario'):
            print(probs)
        single_video_result[keyword] = texts[probs[0].argsort()[-1]]
    print(single_video_result['scerario'])

    submit_json["test_results"].append(single_video_result)

    with open('jzxd_result.json', 'w', encoding='utf-8') as up:
        json.dump(submit_json, up, ensure_ascii=False)

