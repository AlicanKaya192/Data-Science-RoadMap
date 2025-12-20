import os

def rename_all(root_dir):
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Önce dosyaları düzelt
        for name in files:
            new_name = name.replace(" ", "_").replace("?", "_").replace("(", "_").replace(")", "_")
            if new_name != name:
                os.rename(os.path.join(root, name), os.path.join(root, new_name))
        
        # Sonra klasörleri düzelt
        for name in dirs:
            new_name = name.replace(" ", "_").replace("?", "_").replace("(", "_").replace(")", "_")
            if new_name != name:
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                # Eğer yeni isimde bir klasör zaten varsa (çakışma), taşıma yap
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                else:
                    print(f"Uyarı: {new_path} zaten var, atlanıyor.")

rename_all('.')
print("İsim temizliği tamamlandı!")
