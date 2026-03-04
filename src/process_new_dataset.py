import os
import re
import shutil
from collections import defaultdict

def process_new_dataset():
    """
    Process new dataset from dataset_backup to dataset/data with:
    1. Clear existing dataset/data 
    2. Filter valid pairs from dataset_backup
    3. Sequential numbering: Control 1-N, Diabetic N+1-end
    4. Leave dataset_backup unchanged
    """
    
    # Change to project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Paths (now relative to project root)
    backup_control_dir = "dataset_backup/data/control"
    backup_diabetic_dir = "dataset_backup/data/diabetic"  
    target_control_dir = "dataset/data/control"
    target_diabetic_dir = "dataset/data/diabetic"
    
    print("🔍 Processing new dataset from dataset_backup...")
    
    # Check if backup directories exist
    if not os.path.exists(backup_control_dir):
        print(f"❌ ERROR: {backup_control_dir} not found!")
        return
    if not os.path.exists(backup_diabetic_dir):
        print(f"❌ ERROR: {backup_diabetic_dir} not found!")
        return
    
    def find_valid_pairs(directory):
        """Find files that have both L and R images"""
        if not os.path.exists(directory):
            return {}, []
            
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg'))]
        patient_dict = defaultdict(lambda: {'L': [], 'R': []})
        
        for f in files:
            # Updated regex to handle spaces in filenames like "15R IMG..."
            match = re.match(r"(\d+)([LR])[\s\.](.+)", f, re.IGNORECASE)
            if match:
                patient_num = match.group(1)
                eye = match.group(2).upper()
                patient_dict[patient_num][eye].append(f)
        
        # Return only complete pairs (both L and R exist)
        valid_pairs = {}
        orphaned_files = []
        
        for patient_num, eyes in patient_dict.items():
            if eyes['L'] and eyes['R']:
                # Take the first L and R file if multiples exist (sorted)
                valid_pairs[patient_num] = {
                    'L': sorted(eyes['L'])[0],
                    'R': sorted(eyes['R'])[0]
                }
                # Mark extra files as orphaned if there are duplicates
                if len(eyes['L']) > 1:
                    orphaned_files.extend(eyes['L'][1:])
                if len(eyes['R']) > 1:
                    orphaned_files.extend(eyes['R'][1:])
            else:
                # Files without matching pairs are orphaned
                orphaned_files.extend(eyes['L'] + eyes['R'])
        
        return valid_pairs, orphaned_files
    
    # Analyze backup data
    print("📊 Analyzing dataset_backup...")
    control_pairs, control_orphaned = find_valid_pairs(backup_control_dir)
    diabetic_pairs, diabetic_orphaned = find_valid_pairs(backup_diabetic_dir)
    
    print(f"📈 Control - Valid pairs: {len(control_pairs)}, Orphaned: {len(control_orphaned)}")
    print(f"📈 Diabetic - Valid pairs: {len(diabetic_pairs)}, Orphaned: {len(diabetic_orphaned)}")
    
    if control_orphaned:
        print(f"🗑️  Control orphaned: {control_orphaned[:5]}{'...' if len(control_orphaned) > 5 else ''}")
    if diabetic_orphaned:
        print(f"🗑️  Diabetic orphaned: {diabetic_orphaned[:5]}{'...' if len(diabetic_orphaned) > 5 else ''}")
    
    # Find the highest existing control number to avoid overlap
    # Always start fresh - no preserve logic
    control_count = len(control_pairs)
    diabetic_start_num = control_count + 1
    total_pairs = len(control_pairs) + len(diabetic_pairs)
    diabetic_end_num = diabetic_start_num + len(diabetic_pairs) - 1
    
    print(f"\n🔢 Fresh numbering plan (no preservation):")
    print(f"   🔵 Control: 1 to {control_count}")
    print(f"   🔴 Diabetic: {diabetic_start_num} to {diabetic_end_num}")
    
    # Confirmation
    print(f"\n⚠️  This will:")
    print(f"   🧹 Clear all existing data in dataset/data/")
    print(f"   🧹 Clear all existing orphaned files in orphaned_backup/")
    print(f"   📁 Copy {len(control_pairs)} control pairs (renumber 1-{control_count})")
    print(f"   📁 Copy {len(diabetic_pairs)} diabetic pairs (renumber {diabetic_start_num}-{diabetic_end_num})") 
    print(f"   🗑️  Move {len(control_orphaned) + len(diabetic_orphaned)} orphaned files to orphaned_backup/")
    print(f"   📂 Leave dataset_backup/ unchanged")
    
    response = input(f"\n🤔 Proceed with dataset processing? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Operation cancelled.")
        return
    
    # Clear existing dataset/data directories
    print(f"\n🧹 Clearing existing dataset/data and orphaned_backup...")
    
    # Remove and recreate target directories
    if os.path.exists(target_control_dir):
        shutil.rmtree(target_control_dir)
    if os.path.exists(target_diabetic_dir):
        shutil.rmtree(target_diabetic_dir)
        
    # Clear orphaned_backup directory
    orphaned_backup_dir = "orphaned_backup"
    if os.path.exists(orphaned_backup_dir):
        shutil.rmtree(orphaned_backup_dir)
        
    os.makedirs(target_control_dir, exist_ok=True)
    os.makedirs(target_diabetic_dir, exist_ok=True)
    os.makedirs(f"{orphaned_backup_dir}/control", exist_ok=True)
    os.makedirs(f"{orphaned_backup_dir}/diabetic", exist_ok=True)
    
    # Create .gitkeep files
    with open(os.path.join(target_control_dir, '.gitkeep'), 'w') as f:
        f.write('')
    with open(os.path.join(target_diabetic_dir, '.gitkeep'), 'w') as f:
        f.write('')
    
    print(f"✅ Cleared dataset/data and orphaned_backup directories")
    
    # Move orphaned files to backup
    if control_orphaned or diabetic_orphaned:
        print(f"\n🗑️  Moving orphaned files to orphaned_backup/...")
        
        for filename in control_orphaned:
            src = os.path.join(backup_control_dir, filename)
            dst = os.path.join(f"{orphaned_backup_dir}/control", filename)
            if os.path.exists(src):  # Check if file exists before copying
                shutil.copy2(src, dst)
                print(f"   Control: {filename} → orphaned_backup/control/")
        
        for filename in diabetic_orphaned:
            src = os.path.join(backup_diabetic_dir, filename)
            dst = os.path.join(f"{orphaned_backup_dir}/diabetic", filename)
            if os.path.exists(src):  # Check if file exists before copying
                shutil.copy2(src, dst)
                print(f"   Diabetic: {filename} → orphaned_backup/diabetic/")
    
    # Process control pairs (1 to N)
    print(f"\n🔵 Processing control pairs (renumbering 1-{control_count})...")
    new_number = 1
    
    for old_patient_id in sorted(control_pairs.keys(), key=int):
        pair = control_pairs[old_patient_id]
        
        # Extract filename parts
        left_file = pair['L']
        right_file = pair['R']
        
        left_match = re.match(r"\d+L\.(.+)", left_file)
        right_match = re.match(r"\d+R\.(.+)", right_file)
        
        if left_match and right_match:
            left_suffix = left_match.group(1)
            right_suffix = right_match.group(1)
            
            # Create new filenames with sequential numbering
            new_left = f"{new_number}L.{left_suffix}"
            new_right = f"{new_number}R.{right_suffix}"
            
            # Copy files with new names
            src_left = os.path.join(backup_control_dir, left_file)
            dst_left = os.path.join(target_control_dir, new_left)
            src_right = os.path.join(backup_control_dir, right_file)
            dst_right = os.path.join(target_control_dir, new_right)
            
            shutil.copy2(src_left, dst_left)
            shutil.copy2(src_right, dst_right)
            
            print(f"   Control_{old_patient_id} → {new_number} ({left_file} & {right_file})")
            new_number += 1
    
    # Process diabetic pairs (N+1 to end)
    print(f"\n🔴 Processing diabetic pairs ({diabetic_start_num} to {diabetic_end_num})...")
    new_number = diabetic_start_num
    
    for old_patient_id in sorted(diabetic_pairs.keys(), key=int):
        pair = diabetic_pairs[old_patient_id]
        
        # Extract filename parts
        left_file = pair['L']
        right_file = pair['R']
        
        left_match = re.match(r"\d+L\.(.+)", left_file)
        right_match = re.match(r"\d+R\.(.+)", right_file)
        
        if left_match and right_match:
            left_suffix = left_match.group(1)
            right_suffix = right_match.group(1)
            
            # Create new filenames
            new_left = f"{new_number}L.{left_suffix}"
            new_right = f"{new_number}R.{right_suffix}"
            
            # Copy files with new names
            src_left = os.path.join(backup_diabetic_dir, left_file)
            dst_left = os.path.join(target_diabetic_dir, new_left)
            src_right = os.path.join(backup_diabetic_dir, right_file)
            dst_right = os.path.join(target_diabetic_dir, new_right)
            
            shutil.copy2(src_left, dst_left)
            shutil.copy2(src_right, dst_right)
            
            print(f"   Diabetic_{old_patient_id} → {new_number} ({left_file} & {right_file})")
            new_number += 1
    
    # Clear existing masks and related folders if they exist
    print(f"\n🧹 Clearing existing masks and generated data...")
    folders_to_clear = [
        "dataset/masks/control",
        "dataset/masks/diabetic", 
        "dataset/pancreatic_masks/control",
        "dataset/pancreatic_masks/diabetic"
    ]
    
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, '.gitkeep'), 'w') as f:
                f.write('')
            print(f"   Cleared: {folder}")
    
    print(f"\n✅ Dataset processing complete!")
    print(f"📊 Final statistics:")
    print(f"   🔵 Control patients: 1-{control_count} (fresh sequential numbering)")
    print(f"   🔴 Diabetic patients: {diabetic_start_num}-{diabetic_end_num}")
    print(f"   📁 Total valid pairs: {len(control_pairs) + len(diabetic_pairs)}")
    print(f"   🗑️  Orphaned files moved to orphaned_backup/: {len(control_orphaned) + len(diabetic_orphaned)}")
    print(f"   📂 Source (dataset_backup) unchanged")
    print(f"   🧹 All previous data cleared - fresh start")

if __name__ == "__main__":
    process_new_dataset()