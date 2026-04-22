import os
from ultralytics import YOLO

def train_astana():
    """
    Скрипт для дообучения модели YOLOv8 для детекции автомобилей на снимках Астаны.
    Оптимизирован для Windows 11 и GPU NVIDIA RTX 4070.
    """
    # 1. Инициализация модели (используем предобученные веса для детекции)
    # Используем 'yolov8n.pt' вместо 'yolov8n-seg.pt', так как задача — детекция, а не сегментация.
    # Это исправит IndexError в DataLoader, возникший ранее.
    model = YOLO('yolov8n.pt')

    # 2. Запуск обучения с учетом всех спецификаций ТЗ
    results = model.train(
        # Путь к конфигурации датасета
        data='data.yaml', 
        
        # Основные гиперпараметры
        epochs=500,
        imgsz=640,
        batch=16,
        
        # Настройки для стабильности на Windows
        workers=0,  # Обязательно 0 для предотвращения RuntimeError на Windows
        device=0,   # Используем RTX 4070 (id=0)
        
        # Аугментация для спутниковых снимков
        degrees=180.0, # Вращение на любой угол (компенсация пролетов спутника)
        fliplr=0.5,    # Горизонтальное отражение
        flipud=0.5,    # Вертикальное отражение
        mosaic=1.0,    # Мозаичная аугментация для малых объектов (машин)
        
        # Сохранение результатов
        name='astana_cars_project',
        project='runs/detect', # Папка для сохранения всех графиков и весов
        
        # Метрики
        plots=True, # Генерация графиков PR-curve, Confusion Matrix и др.
        
        # Прочие настройки
        exist_ok=True, # Перезаписывать проект, если он уже существует
        pretrained=True
    )

    print("Обучение завершено. Результаты сохранены в папку 'runs/detect/astana_cars_project'")

if __name__ == '__main__':
    # Обертка обязательна для Windows 11 (multiprocessing spawn)
    train_astana()
