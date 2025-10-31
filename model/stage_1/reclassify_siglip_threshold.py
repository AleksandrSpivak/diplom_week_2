"""
Reclassify SigLIP results with new threshold

Читає results_siglip_v2_business.json (threshold=0.000040)
і створює results_siglip_v2_max.json (threshold=0.000030)
на основі збережених scores.
"""

import json
from pathlib import Path

# Конфігурація
INPUT_FILE = 'results/stage_1/results_siglip_v2_business.json'
OUTPUT_FILE = 'results/stage_1/results_siglip_v2_max.json'
NEW_THRESHOLD = 0.000030

def reclassify_with_threshold(input_path, output_path, threshold):
    """Перекласифікація результатів з новим threshold"""
    
    # Завантажити вхідні дані
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Створити нову структуру з перекласифікацією
    new_data = {}
    
    for category, results in data.items():
        new_results = []
        
        for item in results:
            # Перекласифікувати на основі нового threshold
            new_label = 1 if item['score'] > threshold else 0
            
            new_results.append({
                'image': item['image'],
                'score': item['score'],
                'label': new_label
            })
        
        new_data[category] = new_results
    
    # Зберегти результат
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Reclassified results saved to: {output_path}")
    print(f"  New threshold: {threshold}")
    
    # Статистика
    for category, results in new_data.items():
        positive = sum(1 for item in results if item['label'] == 1)
        total = len(results)
        print(f"  {category}: {positive}/{total} positive")

if __name__ == '__main__':
    print("Starting reclassification...")
    reclassify_with_threshold(INPUT_FILE, OUTPUT_FILE, NEW_THRESHOLD)
    print("\nReclassification completed!")
    