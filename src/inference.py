import sys
#sys.path.insert(0, '/content/src/net/')  # Добавляем путь к директории с models.py
#from models import DenseNet121_change_avg  # Импортируем модель из предоставленного файла
#from src.net.models import DenseNet121_change_avg
import torch
import albumentations
import cv2
import numpy as np
import os
from datetime import datetime
import torch.nn as nn
import torchvision
#from torchvision.models import DenseNet121_Weights, DenseNet169_Weights
import torch.nn.functional as F


class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet121_change_avg, self).__init__()
        #self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.densenet121 = torchvision.models.densenet121(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        print("Шаг 1: densenet121")
        x = self.densenet121(x)
        print("Шаг 2: relu")
        x = self.relu(x)
        print("Шаг 3: avgpool")
        x = self.avgpool(x)
        print("Шаг 4: view")
        x = x.view(-1, 1024)
        print("Шаг 5: mlp")
        x = self.mlp(x)
        
        return x

def inference_pipeline(image_path, model_checkpoint, image_size=256):
    """
    Инференс модели для классификации внутричерепных кровоизлияний
    :param image_path: Путь к DICOM-изображению
    :param model_checkpoint: Путь к файлу с весами модели
    :param image_size: Размер входного изображения (по умолчанию 256)
    :return: Словарь с вероятностями для каждого класса
    """
    # 1. Инициализация модели
    model = DenseNet121_change_avg()
    model = model.to('cpu')
    device = torch.device('cpu')
    state = torch.load(model_checkpoint, map_location=device)
    #state = torch.load(model_checkpoint, map_location='cpu', weights_only=True)  # Загрузка на CPU
    new_state = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
    model.load_state_dict(new_state)
    model.eval()

    #print("Ключи в state_dict:", list(new_state['state_dict'].keys()))
    #print("Ключи в модели:", list(model.state_dict().keys()))
    
    # 2. Предобработка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Загрузка в градациях серого
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Преобразование в RGB
    #print("Исходный размер изображения:", image.shape)
    #image = cv2.resize(image, (image_size, image_size))
    #print("Размер после изменения:", image.shape)

    # Трансформации из predict.txt
    transform = albumentations.Compose([
        albumentations.Normalize(
            mean=(0.456, 0.456, 0.456),
            std=(0.224, 0.224, 0.224),
            max_pixel_value=255.0,
            p=1.0
        )
    ])

    image = transform(image=image)['image']
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.FloatTensor(image).unsqueeze(0)
    #print("Размер тензора:", image.shape)
    
    # 3. Инференс
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).numpy()[0]

    # 4. Форматирование результатов
    classes = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}

def ensemble_inference(image_paths, model_checkpoints, image_size=256):
    results = {}
    for image_path in image_paths:
        probabilities = []
        for checkpoint in model_checkpoints:
            # Используем вашу функцию inference_pipeline для каждой модели
            result = inference_pipeline(image_path, checkpoint, image_size)
            probabilities.append(list(result.values()))

        # Усредняем вероятности по всем моделям
        avg_probs = np.mean(probabilities, axis=0)
        classes = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
        results[os.path.basename(image_path)] = {cls: float(prob) for cls, prob in zip(classes, avg_probs)}

    return results

def save_human_readable(results, output_file):
    """
    Сохраняет предсказания в текстовый файл с форматированием
    :param results: Словарь {имя_файла: {класс: вероятность}}
    :param output_file: Путь к выходному файлу
    """
    with open(output_file, 'w') as f:
        f.write(f"{'='*50}\n")
        f.write(f"{'РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ'.center(50)}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        for filename, probs in results.items():
            f.write(f"Изображение: {filename}\n")
            f.write(f"  Наличие кровоизлияния: {probs['any']*100:.1f}%\n")
            f.write("  Детализация:\n")

            for cls in ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']:
                f.write(f"    - {cls.capitalize().ljust(17)}: {probs[cls]*100:.1f}%\n")

            f.write("\n" + "-"*50 + "\n\n")

def print_results(results):
    print(f"\n{'='*50}")
    print(f"{'РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ'.center(50)}\n")
    print(f"{'='*50}\n")
    print(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    for filename, probs in results.items():
        print(f"Изображение: {filename}\n")
        print(f"  Наличие кровоизлияния: {probs['any']*100:.1f}%\n")
        print("  Детализация:\n")

        for cls in ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']:
            print(f"    - {cls.capitalize().ljust(17)}: {probs[cls]*100:.1f}%\n")
        
        print("\n" + "-"*50 + "\n\n")

# Пример использования
if __name__ == '__main__':
    image_dir = "/app/temppng/"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png'))]

    # Пример использования
    checkpoints = [
        "/app/weights/model_epoch_best_0.pth",
        "/app/weights/model_epoch_best_1.pth",
        "/app/weights/model_epoch_best_2.pth",
        "/app/weights/model_epoch_best_3.pth",
        "/app/weights/model_epoch_best_4.pth",
    ]
    result = ensemble_inference(image_paths, checkpoints)
    save_human_readable(result, '/app/output/predictions.txt')
    print_results(result)
