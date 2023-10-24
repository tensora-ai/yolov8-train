import os
import sys
import shutil
import yaml
import logging

from glob import glob
from pathlib import Path

from ultralytics import YOLO


def extract_zip_to_local(src_file):
    dst_path = Path.cwd() / "data"
    dst_path.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(src_file, dst_path)
    return dst_path


def create_dataset_config(path: Path):
    with open(path / "labels" / "labels.txt", "r") as f:
        labels = [l.strip() for l in f.readlines()]
    with open(path / "data.yaml", "w") as f:
        f.write(f"path: {str(path)}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write(f"names: {str(labels)}\n")


def move_files(src: Path, dst: Path, file_types: list):
    if isinstance(file_types, str):
        file_types = [file_types]
    os.makedirs(dst, exist_ok=True)
    files = []
    for file_type in file_types:
        files += glob(str(src / f"*.{file_type}"))
    for file in files:
        shutil.move(file, dst)


def cvat2ultralytics(path: str):
    """Converts a CVAT YOLO 1.1 dataset to the Ultralytics YOLOv8 format.

    Args:
        path: Folder path to the CVAT dataset.
    """
    path = Path(path)

    # Separate data folder into images and labels
    # Merge train, val and test splits into one folder
    assert (path / "obj_train_data").exists(), "train folder not found"
    move_files(path / "obj_train_data", path / "images", ["jpg", "jpeg", "JPG", "JPEG"])
    move_files(path / "obj_train_data", path / "labels", ["txt"])
    assert len(glob(str(path / "images"))) == len(glob(str(path / "labels")))

    # with open(path / "train.txt", "r") as f:
    #     lines = sorted(f.readlines())
    # lines = [re.sub(r"^.*obj_train_data/", "", l) for l in lines]
    # lines = ["images/" + l for l in lines if not l.startswith(".")]
    paths = sorted(glob(str(path / "images" / "*")))
    paths = [p + "\n" for p in paths]
    with open(path / "train.txt", "w") as f:
        f.writelines(paths)
    # os.rmdir(path / "obj_train_data")
    shutil.rmtree(path / "obj_train_data")

    # Rename and move the text file listing the class names
    shutil.move(path / "obj.names", path / "labels" / "labels.txt")
    os.remove(path / "obj.data")


def autosplit(folder, split=(20, 5)):
    with open(folder / "train.txt", "r") as f:
        paths = f.readlines()
    tr = open(folder / "train.txt", "w")
    va = open(folder / "val.txt", "w")
    for i, path in enumerate(paths):
        # p = path.replace(str(folder), "")
        if i % sum(split) >= split[0]:
            va.write(path)
        else:
            tr.write(path)
    tr.close()
    va.close()


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Please provide path to dataset folder and output path"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    src_path = Path(sys.argv[1])
    logging.info(f"Source Path: {src_path}")
    lcl_path = extract_zip_to_local(src_path)
    logging.info(f"Local Path: {lcl_path}")
    logging.info("Extracted dataset to local")

    logging.info(glob(str(lcl_path / "*")))
    dst_path = Path(sys.argv[2])
    logging.info(f"Destination Path: {dst_path}")

    cvat2ultralytics(lcl_path)
    logging.info("Converted dataset to Ultralytics format")

    autosplit(lcl_path)
    logging.info("Split dataset into train and validation set")

    create_dataset_config(lcl_path)
    logging.info("Created dataset config")
    model = YOLO("yolov8n.pt")

    project_name = "default_experiment"
    train_params_file = os.path.join("./", "train_params.yaml")

    if os.path.isfile(train_params_file):
        try:
            logging.info(
                "Starting training with provided training parameters from 'train_params.yaml'."
            )
            with open(train_params_file, "r") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
            project_name = yaml_data.get("project", "custom_experiment")
            model.train(
                cfg=str(train_params_file),
                data=str(lcl_path / "data.yaml"),
                device="cuda",
                project=project_name,
            )
            logging.info("Training completed from file")
        except Exception as e:
            logging.error(
                f"An error occurred during training with parameters from the provided file. Please use a file with correct specifications and name 'train_params.yaml'. Error: {e}"
            )
    else:
        logging.info("Starting training with default parameters")
        model.train(
            data=str(lcl_path / "data.yaml"),
            imgsz=800,
            batch=32,
            epochs=100,
            patience=40,
            workers=0,
            device="cuda",
            project=project_name,
        )
        logging.info("Training completed")

    onnx_path = f"{project_name}/train/weights/best.onnx"
    model.export(format="onnx")

    shutil.copy(onnx_path, dst_path)
    logging.info(
        "Exported trained model successfully as ONNX file to blob container 'models'"
    )
