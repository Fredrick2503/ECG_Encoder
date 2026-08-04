import subprocess
from pathlib import Path
import urllib.request
import zipfile
import shutil
import os
from abc import ABC, abstractmethod

class BaseDownloader(ABC):
    """Abstract Base Class for all dataset downloaders."""
    
    @abstractmethod
    def exists(self) -> bool:
        """Checks if the dataset already exists locally."""
        pass
        
    @abstractmethod
    def download(self) -> Path:
        """Downloads and extracts the dataset, returning the path to the dataset directory."""
        pass


class PTBXLDownloader(BaseDownloader):
    """Downloader for the PTB-XL dataset."""
    
    def __init__(self, download_dir: Path, dataset_name: str = "khyeh0719/ptb-xl-dataset", lightweight: bool = True):
        self.dataset_name = dataset_name
        self.download_dir = Path(download_dir)
        self.lightweight = lightweight

    def exists(self) -> bool:
        """Checks if all required database files exist locally."""
        if self.lightweight:
            return (
                self.download_dir.exists() and
                (self.download_dir / "ptbxl_database.csv").exists() and
                (self.download_dir / "scp_statements.csv").exists() and
                (self.download_dir / "records500/00000/00001_hr.hea").exists()
            )
        return (
            self.download_dir.exists() and
            (self.download_dir / "ptbxl_database.csv").exists() and
            (self.download_dir / "scp_statements.csv").exists() and
            (self.download_dir / "records500/21000/21837_hr.hea").exists()
        )

    def download(self) -> Path:
        """Downloads the PTB-XL dataset from Kaggle or falls back to PhysioNet."""
        if self.exists():
            print(f"PTB-XL dataset already exists at {self.download_dir}")
            self._flatten_nested_dirs()
            return self.download_dir

        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        if self.lightweight:
            print(f"Downloading PTB-XL in LIGHTWEIGHT mode to {self.download_dir}...")
            physionet_base = "https://physionet.org/files/ptb-xl/1.0.3/"
            files_to_download = [
                ("ptbxl_database.csv", "ptbxl_database.csv"),
                ("scp_statements.csv", "scp_statements.csv"),
                ("records500/00000/00001_hr.hea", "records500/00000/00001_hr.hea"),
                ("records500/00000/00001_hr.dat", "records500/00000/00001_hr.dat"),
                ("records100/00000/00001_lr.hea", "records100/00000/00001_lr.hea"),
                ("records100/00000/00001_lr.dat", "records100/00000/00001_lr.dat")
            ]
            import requests
            try:
                for rel_url, rel_path in files_to_download:
                    url = physionet_base + rel_url
                    dest_path = self.download_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if dest_path.exists() and dest_path.stat().st_size > 0:
                        continue
                        
                    print(f"Downloading {rel_path}...")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    with open(dest_path, "wb") as f:
                        f.write(response.content)
                print("Lightweight download complete.")
                return self.download_dir
            except Exception as err:
                print(f"Error during lightweight download: {err}")
                raise RuntimeError(f"Failed to download required files in lightweight mode: {err}") from err

        print(f"Downloading full PTB-XL dataset to {self.download_dir}...")

        # Try downloading via Kaggle first
        import sys
        kaggle_success = False
        kaggle_executable = shutil.which("kaggle")
        if kaggle_executable is None:
            # Check virtual env scripts directory
            venv_kaggle = Path(sys.executable).parent / "kaggle.exe"
            if venv_kaggle.exists():
                kaggle_executable = str(venv_kaggle)
            else:
                venv_kaggle_nix = Path(sys.executable).parent / "kaggle"
                if venv_kaggle_nix.exists():
                    kaggle_executable = str(venv_kaggle_nix)
                    
        if kaggle_executable is not None:
            try:
                print("Attempting download via Kaggle CLI...")
                subprocess.run(
                    [
                        kaggle_executable,
                        "datasets",
                        "download",
                        "-d",
                        self.dataset_name,
                        "-p",
                        str(self.download_dir),
                        "--unzip"
                    ],
                    check=True
                )
                print(f"Successfully downloaded and unzipped {self.dataset_name} via Kaggle.")
                kaggle_success = True
            except Exception as e:
                print(f"Kaggle download failed: {e}. Falling back to PhysioNet...")
        else:
            print("Kaggle CLI not installed or not configured. Falling back to PhysioNet...")

        if not kaggle_success:
            # Fallback to direct download from PhysioNet
            physionet_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-database-1.0.3.zip"
            zip_path = self.download_dir / "ptb-xl-dataset.zip"
            print(f"Downloading from PhysioNet: {physionet_url}")
            
            try:
                import requests
                
                # Check existing size for resume
                existing_size = 0
                if zip_path.exists():
                    existing_size = zip_path.stat().st_size
                    print(f"Found existing partial download of size {existing_size / (1024*1024):.2f} MB")
                
                headers = {}
                if existing_size > 0:
                    headers['Range'] = f'bytes={existing_size}-'
                
                # Send request
                response = requests.get(physionet_url, headers=headers, stream=True, timeout=30)
                
                # Handle status codes
                mode = 'wb'
                if response.status_code == 206:
                    print("Resuming download from last checkpoint...")
                    mode = 'ab'
                    file_size = existing_size + int(response.headers.get('content-length', 0))
                    downloaded = existing_size
                elif response.status_code == 200:
                    if existing_size > 0:
                        print("Server does not support range requests or file has changed. Restarting download...")
                    mode = 'wb'
                    file_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                elif response.status_code == 416:
                    # Requested Range Not Satisfiable: usually means the file is already fully downloaded
                    print("Range not satisfiable. Checking if zip file is valid...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            pass
                        print("Zip file is valid. Skipping download.")
                        file_size = existing_size
                        downloaded = existing_size
                        response = None
                    except zipfile.BadZipFile:
                        print("Zip file is corrupt. Restarting download...")
                        mode = 'wb'
                        response = requests.get(physionet_url, stream=True, timeout=30)
                        file_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                else:
                    response.raise_for_status()
                
                if response is not None:
                    print(f"File size: {file_size / (1024*1024):.2f} MB")
                    chunk_size = 1024 * 1024  # 1MB
                    
                    with open(zip_path, mode) as out_file:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                out_file.write(chunk)
                                downloaded += len(chunk)
                                if int(downloaded) % (50 * chunk_size) == 0 or downloaded == file_size:
                                    print(f"Downloaded {downloaded / (1024*1024):.2f} MB / {file_size / (1024*1024):.2f} MB")

                print("Extracting files...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.download_dir)
                zip_path.unlink()
                print("Extraction completed.")
            except Exception as err:
                print(f"Error during download or extraction: {err}")
                print("Keeping downloaded file for potential resume. Please run the test again to continue.")
                raise RuntimeError(
                    f"Failed to download/extract PTB-XL dataset. You can retry, or download manually from {physionet_url} "
                    f"and extract it to {self.download_dir}"
                ) from err

        # Flatten nested directory structure if any
        self._flatten_nested_dirs()
        return self.download_dir

    def _flatten_nested_dirs(self):
        """Flattens any nested ptb-xl-* directories created during zip extraction."""
        for path in list(self.download_dir.iterdir()):
            if path.is_dir() and "ptb-xl" in path.name.lower():
                print(f"Flattening nested directory: {path.name}")
                for item in path.iterdir():
                    dest = self.download_dir / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(self.download_dir))
                path.rmdir()
