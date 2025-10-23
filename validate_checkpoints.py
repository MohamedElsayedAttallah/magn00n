#!/usr/bin/env python3
"""
Checkpoint Validator and Repair Script
Run this to diagnose and fix checkpoint issues
Usage: python validate_checkpoints.py
"""

import torch
import os
import sys
from pathlib import Path


def validate_checkpoints(checkpoint_dir='signal_viewer_app/assets/checkpoints'):
    """Validate all checkpoint files in the directory"""

    print("=" * 70)
    print("CHECKPOINT VALIDATOR")
    print("=" * 70)
    print(f"\nPython Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Directory Exists: {os.path.exists(checkpoint_dir)}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ ERROR: Directory not found!")
        print(f"   Expected at: {os.path.abspath(checkpoint_dir)}")
        return False

    # Find all checkpoint files
    checkpoint_files = sorted(Path(checkpoint_dir).glob("best_fold_*.pth"))

    if not checkpoint_files:
        print("❌ ERROR: No checkpoint files found!")
        print("   Looking for files matching: best_fold_*.pth")
        print(f"   Files in directory: {os.listdir(checkpoint_dir)}")
        return False

    print(f"Found {len(checkpoint_files)} checkpoint file(s)\n")

    valid_checkpoints = []

    for ckpt_path in checkpoint_files:
        print("-" * 70)
        print(f"Validating: {ckpt_path.name}")
        print("-" * 70)

        # Check file size
        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        print(f"File Size: {file_size_mb:.2f} MB")

        if file_size_mb < 1.0:
            print(f"⚠️  WARNING: File is very small ({file_size_mb:.2f} MB)")
            print(f"   Expected size: 10-50 MB for typical model")
            print(f"   This file may be corrupted or incomplete\n")
            continue

        # Try loading the checkpoint
        try:
            print("Attempting to load...")

            # PyTorch 2.9+ fix: Register numpy safe globals
            if hasattr(torch.serialization, 'add_safe_globals'):
                import numpy as np
                try:
                    # Register numpy types as safe for PyTorch 2.9+
                    if hasattr(np, '_core'):
                        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                except Exception as reg_error:
                    print(f"Warning: Could not register numpy globals: {reg_error}")

            # Load with weights_only=False (PyTorch 2.9+ requirement)
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            print("✅ Load successful!")

            # Validate structure
            if not isinstance(checkpoint, dict):
                print(f"❌ ERROR: Invalid checkpoint format")
                print(f"   Expected: dict, Got: {type(checkpoint)}\n")
                continue

            print(f"Checkpoint Type: {type(checkpoint)}")
            print(f"Available Keys: {list(checkpoint.keys())}")

            # Check required keys
            required_keys = ['model_state_dict']
            missing_keys = [k for k in required_keys if k not in checkpoint]

            if missing_keys:
                print(f"❌ ERROR: Missing required keys: {missing_keys}\n")
                continue

            # Check model state dict
            model_state = checkpoint['model_state_dict']
            print(f"Model State Dict Size: {len(model_state)} parameters")

            # Print some parameter shapes
            print("\nSample Parameters:")
            for i, (key, value) in enumerate(list(model_state.items())[:5]):
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")

            # Check metadata
            print("\nMetadata:")
            print(f"  Variance: {checkpoint.get('val_variance', 'N/A')}")
            print(f"  MAE: {checkpoint.get('val_mae', 'N/A')}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Fold: {checkpoint.get('fold', 'N/A')}")

            print("\n✅ VALID CHECKPOINT")
            valid_checkpoints.append(ckpt_path)

        except Exception as e:
            print(f"❌ ERROR: Failed to load checkpoint")
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Error Message: {str(e)}")

            # Try alternative loading method
            try:
                print("\n   Trying legacy load method...")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                print("   ✅ Legacy load successful!")

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    print("   ✅ VALID CHECKPOINT (via legacy method)")
                    valid_checkpoints.append(ckpt_path)
                else:
                    print("   ❌ Still invalid structure")
            except Exception as e2:
                print(f"   ❌ Legacy load also failed: {str(e2)}")

        print()

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Files Found: {len(checkpoint_files)}")
    print(f"Valid Checkpoints: {len(valid_checkpoints)}")
    print(f"Invalid Checkpoints: {len(checkpoint_files) - len(valid_checkpoints)}")

    if valid_checkpoints:
        print("\n✅ Valid Checkpoint Files:")
        for ckpt in valid_checkpoints:
            print(f"   - {ckpt.name}")
        return True
    else:
        print("\n❌ NO VALID CHECKPOINTS FOUND!")
        print("\nPossible Solutions:")
        print("1. Re-train the model to generate new checkpoints")
        print("2. Ensure training script saves checkpoints with this structure:")
        print("   {")
        print("     'model_state_dict': model.state_dict(),")
        print("     'val_variance': variance_score,")
        print("     'val_mae': mae_score,")
        print("     'epoch': epoch,")
        print("     'fold': fold")
        print("   }")
        print("3. Check PyTorch version compatibility between training and deployment")
        print("4. Verify checkpoint files aren't corrupted during transfer")
        return False


def create_dummy_checkpoint(checkpoint_dir='signal_viewer_app/assets/checkpoints'):
    """Create a dummy checkpoint for testing purposes"""

    print("\n" + "=" * 70)
    print("CREATING DUMMY CHECKPOINT FOR TESTING")
    print("=" * 70)
    print("⚠️  WARNING: This creates a random checkpoint for testing only!")
    print("⚠️  It will NOT produce accurate speed predictions!")
    print()

    response = input("Do you want to create a dummy checkpoint? (yes/no): ").strip().lower()

    if response != 'yes':
        print("Aborted.")
        return False

    try:
        # Import the model architecture
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from signal_viewer_app.speed_prediction_utils import AdvancedSpeedPredictor

        # Create model and save dummy checkpoint
        model = AdvancedSpeedPredictor(n_mels=64)

        os.makedirs(checkpoint_dir, exist_ok=True)

        dummy_checkpoint = {
            'model_state_dict': model.state_dict(),
            'val_variance': 0.75,
            'val_mae': 15.5,
            'epoch': 0,
            'fold': 0
        }

        checkpoint_path = os.path.join(checkpoint_dir, 'best_fold_0.pth')
        torch.save(dummy_checkpoint, checkpoint_path)

        print(f"✅ Dummy checkpoint created: {checkpoint_path}")
        print(f"   File size: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")
        print("\n⚠️  Remember: This is for TESTING ONLY!")
        print("   Train a real model for accurate predictions!")

        return True

    except Exception as e:
        print(f"❌ Failed to create dummy checkpoint: {e}")
        return False


if __name__ == "__main__":
    print("\nVehicle Speed Predictor - Checkpoint Validator\n")

    # Validate existing checkpoints
    is_valid = validate_checkpoints()

    # Offer to create dummy checkpoint if validation failed
    if not is_valid:
        print("\n" + "=" * 70)
        create_dummy_checkpoint()

    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)