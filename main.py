from ultralytics import YOLO

if __name__ == '__main__':
    # 直接使用预测模型创建模型
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml','data':'data.yaml'})

    # 使用yaml配置文件来创建模型，并导入预训练权重
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    model.load('yolov8n.pt')
    model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'data.yaml'})

    # # 模型验证
    # model = YOLO('runs/detect/weights/best.pt')
    # model.val(**{'data':'data.yaml'})
    #
    # # 模型推理
    # model = YOLO('runs/detect/weights/best.pt')
    # model.predict(source='dataset/images/test',**{'save':True})
