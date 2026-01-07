#!/usr/bin/env python3
"""
Quick retrain script for enhanced features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_enhanced_model import train_enhanced_model

if __name__ == "__main__":
    try:
        print("🚀 Retraining model with enhanced angle features...")
        model, encoder, info = train_enhanced_model()
        print("✅ Model retrained successfully!")
        print(f"Classes: {info['classes']}")
        print(f"Accuracy: {info['final_accuracy']:.2%}")
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()