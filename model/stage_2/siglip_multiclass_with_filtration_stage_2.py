"""
SigLIP Multiclass with Filtration

Застосовує threshold-фільтрацію до результатів siglip_multiclass:
- Якщо predicted_label != "Irrelevant" і max_score < threshold → змінюємо на "Irrelevant"
- Зберігає результати у results/stage_2/results_siglip_multiclass_with_filtration.json
"""

import json
from pathlib import Path

class SigLIPFilterProcessor:
    def __init__(self, threshold=0.000010, results_dir='results/stage_2'):
        self.threshold = threshold
        self.results_dir = Path(results_dir)
        self.input_path = self.results_dir / 'results_siglip_multiclass.json'
        self.output_path = self.results_dir / 'results_siglip_multiclass_with_filtration.json'
    
    def load_results(self):
        """Завантажити результати siglip_multiclass"""
        if not self.input_path.exists():
            print(f"Error: {self.input_path} not found!")
            return []
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def apply_filtration(self, results):
        """Застосувати фільтрацію через threshold"""
        filtered_results = []
        changes_count = 0
        
        for result in results:
            image = result['image']
            true_label = result['true_label']
            predicted_label = result['predicted_label']
            scores = result['scores']
            
            # Якщо вже Irrelevant - пропускаємо
            if predicted_label == "Irrelevant":
                filtered_results.append(result)
                continue
            
            # Для інших класів перевіряємо score
            predicted_score = scores.get(predicted_label, 0)
            
            # Якщо score < threshold → змінюємо на Irrelevant
            if predicted_score < self.threshold:
                filtered_results.append({
                    'image': image,
                    'true_label': true_label,
                    'predicted_label': 'Irrelevant',
                    'scores': scores,
                    'original_prediction': predicted_label,
                    'filtered': True
                })
                changes_count += 1
            else:
                filtered_results.append(result)
        
        return filtered_results, changes_count
    
    def save_results(self, results):
        """Збереження результатів"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def process(self):
        """Обробка результатів"""
        print(f"Loading results from: {self.input_path}")
        results = self.load_results()
        
        if not results:
            return
        
        print(f"Total images: {len(results)}")
        print(f"Applying threshold filtration (threshold={self.threshold})...")
        
        filtered_results, changes_count = self.apply_filtration(results)
        
        print(f"✓ Filtration applied: {changes_count} predictions changed to Irrelevant")
        
        self.save_results(filtered_results)
        
        print(f"✓ Results saved to: {self.output_path}")
        
        return filtered_results

if __name__ == '__main__':
    print("Starting SigLIP Multiclass Filtration...")
    
    processor = SigLIPFilterProcessor(threshold=0.000010)
    results = processor.process()
    
    print("\nFiltration completed!")