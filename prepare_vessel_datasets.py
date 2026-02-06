"""
Prepare and consolidate vessel detection datasets for YOLOv8 training.
Combines datasets from archive/v1-v7 and Maritime YOLO.v1i.yolov8
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import hashlib
from tqdm import tqdm


def get_image_hash(image_path):
    """Get hash of image file to detect duplicates."""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def read_yaml_config(yaml_path):
    """Read YOLO dataset YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def copy_files_with_renaming(source_dir, dest_dir, prefix="", existing_hashes=None):
    """
    Copy files from source to destination, renaming to avoid conflicts.
    Returns dict mapping original names to new names and set of file hashes.
    """
    if existing_hashes is None:
        existing_hashes = set()
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    name_mapping = {}
    new_hashes = set()
    
    source_path = Path(source_dir)
    if not source_path.exists():
        return name_mapping, existing_hashes
    
    files = list(source_path.glob("*.*"))
    for file_path in tqdm(files, desc=f"Copying {prefix}"):
        # Get file hash to check for duplicates
        file_hash = get_image_hash(file_path)
        
        if file_hash in existing_hashes:
            # Skip duplicate
            continue
        
        # Create new filename with prefix
        new_name = f"{prefix}_{file_path.name}" if prefix else file_path.name
        dest_path = dest_dir / new_name
        
        # Handle name conflicts
        counter = 1
        while dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.copy2(file_path, dest_path)
        name_mapping[file_path.name] = dest_path.name
        existing_hashes.add(file_hash)
        new_hashes.add(file_hash)
    
    return name_mapping, existing_hashes


def update_label_file(label_path, class_mapping, name_mapping, new_label_path):
    """Update YOLO label file with new class IDs and verify image name matches."""
    if not label_path.exists():
        return False
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            old_class_id = int(parts[0])
            # Map to new class ID
            new_class_id = class_mapping.get(old_class_id, old_class_id)
            parts[0] = str(new_class_id)
            updated_lines.append(' '.join(parts) + '\n')
        
        # Write updated label
        new_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_label_path, 'w') as f:
            f.writelines(updated_lines)
        
        return True
    except Exception as e:
        print(f"Error updating label {label_path}: {e}")
        return False


def consolidate_datasets():
    """Main function to consolidate all vessel datasets."""
    
    base_dir = Path("datasets")
    archive_dir = base_dir / "archive"
    maritime_dir = base_dir / "Maritime YOLO.v1i.yolov8"
    output_dir = base_dir / "combined"
    
    # Create output directories
    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    valid_images_dir = output_dir / "valid" / "images"
    valid_labels_dir = output_dir / "valid" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, valid_images_dir, valid_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Track file hashes to avoid duplicates
    image_hashes = set()
    
    # Class mapping: We'll use a single "vessel" class (class 0)
    # Archive datasets use class 0, Maritime uses multiple classes - we'll map all to 0
    unified_classes = {"vessel": 0}
    class_mapping = {}  # Will be populated per dataset
    
    print("=" * 60)
    print("Consolidating Vessel Detection Datasets")
    print("=" * 60)
    
    # Process archive datasets v1 through v7
    archive_datasets = []
    for v_num in range(1, 8):  # v1 through v7
        v_dir = archive_dir / f"v{v_num}"
        if not v_dir.exists():
            print(f"Warning: {v_dir} does not exist, skipping")
            continue
        
        yaml_path = v_dir / "data.yaml"
        if not yaml_path.exists():
            print(f"Warning: {yaml_path} does not exist, skipping v{v_num}")
            continue
        
        archive_datasets.append((v_num, v_dir, yaml_path))
    
    print(f"\nFound {len(archive_datasets)} archive datasets (v1-v7)")
    
    # Process each archive dataset
    for v_num, v_dir, yaml_path in archive_datasets:
        print(f"\nProcessing archive/v{v_num}...")
        config = read_yaml_config(yaml_path)
        
        # Archive datasets use class 0 for vessel
        class_mapping = {0: 0}  # Map class 0 to class 0 (vessel)
        
        # Process training data
        train_img_dir = v_dir / "train" / "images"
        train_lbl_dir = v_dir / "train" / "labels"
        
        if train_img_dir.exists():
            img_mapping, image_hashes = copy_files_with_renaming(
                train_img_dir, train_images_dir, prefix=f"v{v_num}", existing_hashes=image_hashes
            )
            
            # Copy corresponding labels
            for orig_name, new_name in img_mapping.items():
                label_name = Path(orig_name).stem + ".txt"
                orig_label = train_lbl_dir / label_name
                new_label = train_labels_dir / f"{Path(new_name).stem}.txt"
                
                if orig_label.exists():
                    update_label_file(orig_label, class_mapping, img_mapping, new_label)
        
        # Process validation data
        valid_img_dir = v_dir / "valid" / "images"
        valid_lbl_dir = v_dir / "valid" / "labels"
        
        if valid_img_dir.exists():
            img_mapping, image_hashes = copy_files_with_renaming(
                valid_img_dir, valid_images_dir, prefix=f"v{v_num}", existing_hashes=image_hashes
            )
            
            # Copy corresponding labels
            for orig_name, new_name in img_mapping.items():
                label_name = Path(orig_name).stem + ".txt"
                orig_label = valid_lbl_dir / label_name
                new_label = valid_labels_dir / f"{Path(new_name).stem}.txt"
                
                if orig_label.exists():
                    update_label_file(orig_label, class_mapping, img_mapping, new_label)
    
    # Process Maritime YOLO dataset
    print(f"\nProcessing Maritime YOLO.v1i.yolov8...")
    maritime_yaml = maritime_dir / "data.yaml"
    
    if maritime_yaml.exists():
        config = read_yaml_config(maritime_yaml)
        
        # Maritime dataset has 5 classes: anchor, boat, buoy, lighthouse, ship
        # Map all vessel-related classes to class 0 (vessel)
        # Classes: 0=anchor, 1=boat, 2=buoy, 3=lighthouse, 4=ship
        # We'll map boat (1) and ship (4) to vessel (0)
        # Optionally keep others or map all to vessel - mapping all to vessel for simplicity
        class_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # All map to vessel
        
        # Process training data
        train_img_dir = maritime_dir / "train" / "images"
        train_lbl_dir = maritime_dir / "train" / "labels"
        
        if train_img_dir.exists():
            img_mapping, image_hashes = copy_files_with_renaming(
                train_img_dir, train_images_dir, prefix="maritime", existing_hashes=image_hashes
            )
            
            for orig_name, new_name in img_mapping.items():
                label_name = Path(orig_name).stem + ".txt"
                orig_label = train_lbl_dir / label_name
                new_label = train_labels_dir / f"{Path(new_name).stem}.txt"
                
                if orig_label.exists():
                    update_label_file(orig_label, class_mapping, img_mapping, new_label)
        
        # Process validation data
        valid_img_dir = maritime_dir / "valid" / "images"
        valid_lbl_dir = maritime_dir / "valid" / "labels"
        
        if valid_img_dir.exists():
            img_mapping, image_hashes = copy_files_with_renaming(
                valid_img_dir, valid_images_dir, prefix="maritime", existing_hashes=image_hashes
            )
            
            for orig_name, new_name in img_mapping.items():
                label_name = Path(orig_name).stem + ".txt"
                orig_label = valid_lbl_dir / label_name
                new_label = valid_labels_dir / f"{Path(new_name).stem}.txt"
                
                if orig_label.exists():
                    update_label_file(orig_label, class_mapping, img_mapping, new_label)
    
    # Create unified data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 1,  # Number of classes
        'names': ['vessel']  # Class names
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    # Print statistics
    train_images = len(list(train_images_dir.glob("*.*")))
    train_labels = len(list(train_labels_dir.glob("*.txt")))
    valid_images = len(list(valid_images_dir.glob("*.*")))
    valid_labels = len(list(valid_labels_dir.glob("*.txt")))
    
    print("\n" + "=" * 60)
    print("Dataset Consolidation Complete!")
    print("=" * 60)
    print(f"Training images: {train_images}")
    print(f"Training labels: {train_labels}")
    print(f"Validation images: {valid_images}")
    print(f"Validation labels: {valid_labels}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Data YAML: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    consolidate_datasets()
