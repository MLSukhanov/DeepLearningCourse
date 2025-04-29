import os
import sys
import time
import numpy as np
from PIL import Image

# Проверка PNG-файлов
def validate_png_files(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return False
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Ошибка в файле {filename}: {e}")
                return False
    return True

# Парсинг файлов отчета predictions.txt
def parse_predictions(path):
    results = {}
    current = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Изображение:'):
                current = line.split('Изображение:')[1].strip()
                results[current] = {}
            elif 'Наличие кровоизлияния:' in line and current:
                val = float(line.split(':')[1].strip().rstrip('%')) #/ 100.0
                results[current]['any'] = val
            elif line.startswith('-') and ':' in line and current:
                # Убираем ведущий '-', сплитуем только по первому ':' и игнорируем строки без ':' 
                parts = line.lstrip('-').split(':', 1)
                cls = parts[0].strip().lower()
                val_str = parts[1].strip().rstrip('%')
                # На всякий случай проверяем, что после ':' есть число
                if val_str:
                    val = float(val_str) #/ 100.0
                    results[current][cls] = val
    return results

if __name__ == '__main__':
    # Замер времени предобработки
    start_pre = time.perf_counter()
    status_pre = os.system('python /app/src/preprocess.py')
    end_pre = time.perf_counter()
    t_pre_ms = (end_pre - start_pre) * 1000

    if status_pre != 0:
        print('Ошибка предобработки DICOM-файлов')
        sys.exit(1)

    # Проверка PNG
    temp_dir = '/app/temppng'
    if not validate_png_files(temp_dir):
        print('Обнаружены некорректные PNG-файлы')
        sys.exit(1)
    print('Предобработка медицинских изображений выполнена успешно\n')

    # Сравнение количества PNG до запуска инференса
    etalon_dir = '/app/etalon'
    png_etalon = [f for f in os.listdir(etalon_dir) if f.lower().endswith('.png')]
    png_temp = [f for f in os.listdir(temp_dir) if f.lower().endswith('.png')]
    if len(png_etalon) != len(png_temp):
        print('Ошибка: количество изображений в etalon и temppng различается, инференс не будет запущен')
        sys.exit(1)

    # Сравнение изображений
    print('Сравнение изображений:')
    epsilon = 0.00001  # Эпсилон для сравнения вещественных значений
    for fname in sorted(png_etalon):
        if fname in png_temp:
            img_e = np.array(Image.open(os.path.join(etalon_dir, fname)).convert('L'), dtype=np.float32)
            img_t = np.array(Image.open(os.path.join(temp_dir, fname)).convert('L'), dtype=np.float32)
            total = img_e.size
            diff_mask = np.abs(img_e - img_t) <= epsilon
            diff_count = np.count_nonzero(diff_mask)
            sum_diff = np.abs(img_e - img_t).sum()
            print(f'{fname}: совпадающих пикселей {diff_count}, всего пикселей {total}, сумма разниц {sum_diff:.5f}')

    # Замер времени инференса
    start_inf = time.perf_counter()
    status_inf = os.system('python /app/src/inference.py')
    end_inf = time.perf_counter()
    t_inf_s = end_inf - start_inf

    if status_inf != 0:
        print('Ошибка выполнения классификатора')
        sys.exit(1)
    print('Инференс успешно выполнен\n')

    # Вывод и сохранение времени
    out_dir = '/app/output'
    os.makedirs(out_dir, exist_ok=True)
    time_file = os.path.join(out_dir, 'time.txt')
    with open(time_file, 'w') as tf:
        tf.write(f'preprocess_time_ms: {t_pre_ms:.2f}\n')
        tf.write(f'inference_time_s: {t_inf_s:.2f}\n')
    print(f'Время предобработки: {t_pre_ms:.2f} мс')
    print(f'Время инференса: {t_inf_s:.2f} с')

    # Сравнение predictions.txt
    etalon_preds = parse_predictions(os.path.join(etalon_dir, 'predictions.txt'))
    output_preds = parse_predictions(os.path.join(out_dir, 'predictions.txt'))
    print('Сравнение результатов классификации:')
    for img_name, e_vals in etalon_preds.items():
        if img_name in output_preds:
            o_vals = output_preds[img_name]
            diff_any = o_vals['any'] - e_vals['any']
            print(f'{img_name}: разница ANY = {diff_any:.4f}')
            for cls in ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']:
                d = abs(o_vals.get(cls,0.0) - e_vals.get(cls,0.0))
                print(f'  {cls}: abs diff = {d:.4f}')

    # Сравнение времени с etalon/time.txt
    dev_pre = dev_inf = None
    with open(os.path.join(etalon_dir, 'time.txt')) as tf:
        for line in tf:
            if 'preprocess_time_ms' in line:
                dev_pre = float(line.split(':')[1])
            elif 'inference_time_s' in line:
                dev_inf = float(line.split(':')[1])
    print('Сравнение времени выполнения:')
    print(f'  Разработчик: preprocess {dev_pre:.2f} мс, inference {dev_inf:.2f} с')
    print(f'  Пользователь: preprocess {t_pre_ms:.2f} мс, inference {t_inf_s:.2f} с')
