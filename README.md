# CUDAGravity

This is a slightly dodgy gravity simulator written using CUDA.

This project contains three different applications:

## `cudagravity_sim`

This is the main simulator, it expects an `initial.ibin` file in it's current directory and generates `<epoch number>.bin` files 
in the `output` directory it creates, it is the only application that requires CUDA.

## `cudagravity_viewer`

This is the viewer application, it takes the `<epoch number>`.bin files that `cudagravity_sim` creates and shows them on the screen via SDL. 

Currently these are the available keys:

| Key   | Function |
|-------|----------|
| <kbd>space</kbd> | Pauses/unpauses the viewer (starts paused) |
| <kbd>a</kbd> | Turns auto-size on/off (auto size keeps every particle in the viewport |
| <kbd>&#8594;</kbd>/<kbd>&#8592;</kbd> | Moves the viewer's time forwards/backwards by `dt` |
| <kbd>+</kbd>/<kbd>-</kbd> | *On the numpad*, Increases or decreases `dt` by a factor of 2 | 
| <kbd>0</kbd> | *On the numpad*, Goes back to the start |
| <kbd>↑</kbd>/<kbd>↓</kbd> | Zooms in/out, if you try zoom in past a particle with autosize on it won't do anything (use <kbd>SHIFT</kbd> for a bigger zoom interval and <kbd>CTRL</kbd> for a massive zoom interval |

## `generate_initial.py`

This generates the `initial.ibin` file required by `cudagravity_sim`, it has three modes:

### Random (`-r N`)
In this mode you pass in the number of particles you want and it generates `N` random particles and writes those to a file

### YAML (`-f filename`)
This reads the yaml file that you specify to generate the initial file (see [generate_initial/sample.yaml](generate_initial/sample.yaml)) for an example

### Editor (no arguments specified)
This opens a pygame based editor application, it's very simple but it works, you just click to create a particle and add velocity to it by dragging your mouse cursor
