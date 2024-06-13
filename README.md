# Human Pose Keypoint Annotation Tool


## Setup
Clone this repository.
```
git clone https://github.com/naoya1110/pose_annotation_tool.git
```

Navigate to the cloned directory.
```
cd pose_annotation_tool
```

Create a virtual environment.
```
python3 -m venv .venv
```

Install required packages.
```
pip install ultralytics flet
```

## Run in Ubuntu
```
source .venv/bin/activate
python3 main.py
```

## Trouble Shooting
Flet filepicker may not work in VScode with the default settings. You might be able to fix this issue by changing `settings.json` in VScode.
```
"terminal.integrated.env.linux": {
    "GTK_PATH": null,
    "GIO_MODULE_DIR": null,
},
```
