"""
/terminals/1
CUSTOM DATASET: /workspace/yolo11/datasets/intra/C0021/data.yaml
OPENTTGAME DATASET: /workspace/yolo11/datasets/opentt_game/openttgames.yaml
""" 

import ultralytics
from ultralytics import YOLO

# =============================
# -------YOLOs version---------
# =============================
# model=YOLO("yolo12n.yaml", verbose=True) 
# model=YOLO("yolo11n.yaml", verbose=True) 
# model=YOLO("yolov10n.yaml", verbose=True)
model=YOLO("yolov9t.yaml", verbose=True) 
# model=YOLO("leyolonano.yaml", verbose=True) 
# model=YOLO("yolov8n.yaml", verbose=True) 

# ================================================================
# =====================REMEMBER CHANGE MISH AND SIOU==============
# ================================================================
model.train(
                    data="/workspace/datasets/OpenTTGame/openttgames.yaml",
                    # data="/workspace/datasets/WASNLab/data.yaml", 
                    epochs=2,                     
                    imgsz=1280,                    
                    batch=16,                      
                    device=[0,1],                  
                    workers=16,
                    
                    # ---- OPTIMIZER & LR SCHEDULE ----
                    optimizer="AdamW",             
                    lr0=0.0007,
                    lrf=0.01,
                    weight_decay=0.0005,                   
                    warmup_epochs=3,               
                    momentum=0.9,                 
                    
                    # ---- TRAINING BEHAVIOR ----
                    patience=15,  
                    mosaic=0.0,
                    mixup=0.0,
                    degrees=0.0,
                    translate=0.1,
                    scale=0.6,
                    fliplr=0.0,
                    project="/workspace/yolo11/runs",
                    name="YOLOv9t_100epoch_32batch_openttgame")

metrics=model.val(
    data="/workspace/datasets/OpenTTGame/openttgames.yaml",
    imgsz=1280,
    # imgsz=960,
    batch=32,
    device=[0,1]
)

print('Train YOLOv9 on OpenTTGame - 100 epochs imgsz=1280')
print("Precision=",metrics.box.p)         # precision
print("Recal=",metrics.box.r)         # recall
print("mAP@0.5=",metrics.box.map50)     # mAP@0.5
print("mAP@0.5:0.95=", metrics.box.map)       # mAP@0.5:0.95
