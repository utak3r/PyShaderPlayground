# PyShaderPlayground

PyShaderPlayground is a simple tool for editing GLSL fragment shaders and instantly testing them in an OpenGL window. There're few predefined uniforms, like image resolution or running time. The names of those are made to be compatible with *ShaderToy*. The editor has a simple syntax highlighting.

It also has offscreen rendering implemented, to render high resolution images, also it's capable of rendering an animation of a given resolution, fps and length. It's utilizing ffmpeg for that task.

![](docs/screenshot.png)

## Requirements

PyShaderPlayground is written in *Python 3.12* and *Qt* (*PySide6*).

* Python 3.12.6
* PySide6 6.7.3
* PyOpenGL 3.1.7
* scipy 1.14.1
* matplotlib 3.9.2
* scikit-image 0.24.0

There're also some Visual Code files included (_launch.json_ and _tasks.json_)

## Creating standalone app ##

There's a possibility to use PyInstaller to create a standalone application, 
outside of Python environment (also for Windows).
There's a proper _spec_ file included, which defines all needed things in order
to achieve that. Just run:

```
pyinstaller --clean PyShaderPlayground.spec
```

or use predefined task in Visual Code.
