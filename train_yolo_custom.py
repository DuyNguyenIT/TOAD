# /terminals/2
import ultralytics
from ultralytics import YOLO
import torch

# model=YOLO("yolov10n.yaml", verbose=True)
model=YOLO("yolo11_custom3.yaml", verbose=True)
# model=YOLO("yolo12.yaml", verbose=True)
# model=YOLO("yolov9t.yaml", verbose=True)
# model=YOLO("yolo11.yaml",  verbose=True)
# model=YOLO("yolov9_custom.yaml", verbose=True)
# model=YOLO("/workspace/yolo11/runs/YOLOv11_custom3_100epoch_openttgame3/weights/best.pt")

results=model.train(
                    # data="/workspace/datasets/WASNLab/data.yaml", 
                    # data="/workspace/datasets/OpenTTGame/openttgames.yaml",
                    epochs=100,
                    # imgsz=1280,
                    imgsz=960,
                    optimizer='AdamW',
                    weight_decay=0.0045,
                    lr0=0.001,
                    momentum=0.937,
                    # momentum=0.8,
                    batch=32,
                    workers=8,
                    patience= 15,
                    device=[0,1],
                    project="/workspace/runs",
                    name="yolov11_small_custom_100epoch_32batch",
                    warmup_epochs=4
        )

metrics=model.val(
    # data="/workspace/datasets/WASNLab/data.yaml",
    # data="/workspace/datasets/OpenTTGame/openttgames.yaml",
    data="/workspace/datasets/drop_ball/drop_ball.yaml",
    # imgsz=1280,
    imgsz=960,
    batch=32,
    device=[0,1]
)

print("Precision=",metrics.box.p)         # precision
print("Recal=",metrics.box.r)         # recall
print("mAP@0.5=",metrics.box.map50)     # mAP@0.5
print("mAP@0.5:0.95=", metrics.box.map)       # mAP@0.5:0.95
