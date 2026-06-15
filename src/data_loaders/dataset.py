# /home/elgin-dev/medifairnet/src/data_loaders/dataset.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class MediFairXRayDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, disease_cols: list, demographic_col: str):
        self.img_dir = img_dir
        self.disease_cols = disease_cols
        self.demographic_col = demographic_col
        
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            
            # 🧼 Fix column keys immediately upon loading the dataframe
            # This maps "Image Index" to "Image_Index" so the code remains clean!
            if 'Image Index' in self.df.columns:
                self.df = self.df.rename(columns={'Image Index': 'Image_Index'})
            
            self.df['Image_Index'] = self.df['Image_Index'].astype(str).str.strip()
            
            # Dynamic multi-label disease mapping
            if 'Finding Labels' in self.df.columns:
                for disease in disease_cols:
                    self.df[disease] = self.df['Finding Labels'].apply(lambda x: 1 if disease in str(x) else 0)
            
            # Map Patient Gender to 0 or 1
            if 'Patient Gender' in self.df.columns:
                self.df['demographic_group_idx'] = self.df['Patient Gender'].map({'M': 0, 'F': 1}).fillna(0).astype(int)
            else:
                self.df['demographic_group_idx'] = 0
                
            print(f"✅ Filter Success! Unified manifest headers for {len(self.df)} data elements.")
        else:
            print(f"⚠️ Manifest path {csv_file} not found. Running backup simulation engine.")
            self.df = self._create_mock_dataframe(disease_cols)

        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # This key is now explicitly guaranteed to exist!
        img_name = row['Image_Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback if a specific image file isn't downloaded yet
            image = Image.new('RGB', (224, 224), color='gray')
            
        image_tensor = self.transform(image)
        disease_targets = torch.tensor(row[self.disease_cols].values.astype('float32'))
        demographic_target = torch.tensor(int(row['demographic_group_idx']))
        
        active_diseases = [col for col in self.disease_cols if row[col] == 1]
        text_description = f"Chest radiograph presenting indicators of {', '.join(active_diseases)}." if active_diseases else "Normal chest radiograph."
            
        return image_tensor, disease_targets, demographic_target, text_description

    def _create_mock_dataframe(self, disease_cols):
        data = {
            'Image_Index': [f"sample_{i}.png" for i in range(20)],
            'demographic_group_idx': [i % 2 for i in range(20)]
        }
        for col in disease_cols:
            data[col] = [0 for _ in range(20)]
        return pd.DataFrame(data)