import zipfile, os
zip_path = r"C:/Users/Admin/.gemini/antigravity/brain/1e52eeb2-3ccb-45dd-8640-2eea6a439e11/.user_uploaded/media_1793562467402_6e69ac29.zip"
extract_to = r"C:/Users/Admin/Downloads/Anemia-detection/Anemia-detection-main/labels"
os.makedirs(extract_to, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print(f"Extracted {zip_path} to {extract_to}")

