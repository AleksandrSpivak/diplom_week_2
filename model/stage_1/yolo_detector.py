"""
YOLO Detector for Business-Relevant Object Detection

Призначення:
- Використовує pretrained YOLOv8 модель для детекції об'єктів на зображеннях
- Класифікує зображення як релевантні/нерелевантні для бізнес-категорій
- Виявляє об'єкти з COCO dataset (обмежений набір класів)
- Працює з датасетом data/stage_1/ (4 категорії)
- Зберігає результати у results/results_yolo.json

Структура:
- YOLODetector: основний клас для object detection
- detect_objects(): детектує об'єкти на зображенні
- is_relevant(): визначає релевантність для категорії
- process_dataset(): обробляє весь датасет
"""

from ultralytics import YOLO
import os
from pathlib import Path
import json
from tqdm import tqdm

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt', model_path='weights/yolo'):
        """Ініціалізація YOLOv8 моделі"""
        # Створити папку для ваг
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Шлях до моделі
        model_file = model_dir / model_name
        
        # Завантажити модель (автоматично скачається якщо відсутня)
        self.model = YOLO(str(model_file) if model_file.exists() else model_name)
        
        # Якщо модель скачалася, перемістити у weights/yolo
        if not model_file.exists() and Path(model_name).exists():
            Path(model_name).rename(model_file)
        
        # Завантаження категорій з JSON
        with open('available_business_categories.json', 'r', encoding='utf-8') as f:
            all_categories = json.load(f)
        
        # Вибрати тільки 4 категорії
        self.business_categories = {
            "Pet-Friendly Services": all_categories["Pet-Friendly Services"],
            "Catering": all_categories["Catering"],
            "Marine_Activities": all_categories["Marine_Activities"],
            "Cultural_Excursions": all_categories["Cultural_Excursions"]
        }
        
        self.output_path = 'results/stage_1/results_yolo.json'
    
    def detect_objects(self, image_path):
        """Детекція об'єктів на фото"""
        results = self.model(image_path, verbose=False)
        detected_classes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detected_classes.append(class_name)
        
        return list(set(detected_classes))
    
    def is_relevant(self, detected_classes, category):
        """Перевірка релевантності фото для категорії"""
        relevant_objects = self.business_categories.get(category, [])
        if not relevant_objects:
            return 0
        
        # Приводимо до lowercase для порівняння
        relevant_objects_lower = [obj.lower() for obj in relevant_objects]
        detected_classes_lower = [obj.lower() for obj in detected_classes]
        
        for obj in detected_classes_lower:
            if obj in relevant_objects_lower:
                return 1
        return 0
    
    def save_results(self, results):
        """Збереження результатів"""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def process_dataset(self, data_dir='data/stage_1'):
        """Обробка всього датасету"""
        results = {}
        
        for category in self.business_categories.keys():
            category_path = Path(data_dir) / category
            if not category_path.exists():
                print(f"Skipping {category} - directory not found")
                continue
            
            print(f"\nProcessing category: {category}")
            category_results = []
            image_files = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            
            for img_path in tqdm(image_files, desc=f"{category}", unit="img"):
                detected = self.detect_objects(str(img_path))
                label = self.is_relevant(detected, category)
                
                category_results.append({
                    'image': img_path.name,
                    'detected_objects': detected,
                    'label': label
                })
                
                # Оновлюємо results та зберігаємо JSON після кожного фото
                results[category] = category_results
                self.save_results(results)
            
            print(f"Completed {category}: {len(category_results)} images processed")
        
        return results

if __name__ == '__main__':
    print("Starting YOLO detection...")
    detector = YOLODetector()
    results = detector.process_dataset()
    print("\nYOLO detection completed!")
    print(f"Results saved to: {detector.output_path}")