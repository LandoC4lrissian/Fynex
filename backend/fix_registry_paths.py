"""
Fix duplicate backend path in registry.json
Run this after manually moving files from backend/backend/ml to backend/ml
"""
import json
from pathlib import Path

def fix_registry_paths():
    """Fix all model paths in registry.json"""

    registry_file = Path("backend/ml/saved_models/registry.json")

    if not registry_file.exists():
        print(f"❌ Registry file not found: {registry_file}")
        return

    # Load registry
    with open(registry_file, 'r') as f:
        registry = json.load(f)

    # Fix paths
    changes = []
    for model_name, model_data in registry['models'].items():
        for version, version_data in model_data['versions'].items():
            old_path = version_data.get('model_path', '')

            # Fix duplicate backend\\backend\\ -> backend\\
            if 'backend\\backend\\ml' in old_path:
                new_path = old_path.replace('backend\\backend\\ml', 'backend\\ml')
                version_data['model_path'] = new_path
                changes.append({
                    'model': model_name,
                    'version': version,
                    'old': old_path,
                    'new': new_path
                })

    if not changes:
        print("✅ No path changes needed")
        return

    # Show changes
    print(f"\n🔍 Found {len(changes)} path(s) to fix:\n")
    for change in changes:
        print(f"  {change['model']} ({change['version']}):")
        print(f"    Old: {change['old']}")
        print(f"    New: {change['new']}")
        print()

    # Confirm
    response = input("❓ Apply these changes? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("❌ Changes cancelled")
        return

    # Save updated registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"\n✅ Updated {len(changes)} path(s) in registry.json")
    print(f"   Registry saved to: {registry_file}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Registry Path Fixer")
    print("="*60 + "\n")

    fix_registry_paths()

    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60 + "\n")
