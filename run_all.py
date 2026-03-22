import subprocess, sys, os

if __name__ == "__main__":
    print("=" * 45)
    print("  RideIQ — Running training pipeline")
    print("=" * 45)

    result = subprocess.run([sys.executable, "src/train.py"])

    if result.returncode != 0:
        print("Training failed. Check errors above.")
        sys.exit(1)

    print()
    print("=" * 45)
    print("  Done! Launch the app with:")
    print("  python -m streamlit run src/app.py")
    print("=" * 45)