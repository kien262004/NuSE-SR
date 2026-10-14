"""Repository entry point for fitting the PCA decoder."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from nullspaceSR.fit_pca import main

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
