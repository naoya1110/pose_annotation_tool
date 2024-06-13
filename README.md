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

Install required packages in the virtual environment.
```
source .venv/bin/activate
pip install ultralytics flet
```

You might need to install `libmpv.so.1` as well
```
sudo apt update
sudo apt install libmpv-dev
```

## Run main.py
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
Ref: https://askubuntu.com/questions/1462295/ubuntu-22-04-both-eye-of-gnome-and-gimp-failing-with-undefined-symbol-error
