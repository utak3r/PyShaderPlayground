from PySide2.QtWidgets import QOpenGLWidget
from PySide2.QtGui import QOpenGLShader, QOpenGLShaderProgram
from OpenGL import GL as gl
from PySide2.QtCore import QTimer

class ShaderWidget(QOpenGLWidget):
    def __init__(self, width: int, height: int, parent=None):
        super().__init__(parent)
        self.width_ = width
        self.height_ = height
        self.vao_ = None
        self.program_ = None
        self.shader_vertex_ = None
        self.shader_fragment_ = None
        self.attrib_position = None
        self.uniform_iResolution = None
        self.uniform_iGlobalTime = None
        self.global_time: float = 0.0
        self.shader_template_pre_ = ""
        self.shader_template_post_ = ""
        self.shader_user_ = \
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n" \
            "{\n" \
            "vec2 uv = fragCoord.xy / iResolution.xy;\n" \
            "fragColor = vec4(uv,0.5+0.5*sin(iGlobalTime.x),1.0);\n" \
            "}\n"
        self.shader_fallback_ = ""
        
        self.vertices_ = [ -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ]
        self.resize(self.width_, self.height_)

        self.timer_ = QTimer(self)
        self.timer_.timeout.connect(self.timer_tick)
        self.timer_.setInterval(1000 / 50)
        self.timer_.start()

    def timer_tick(self):
        """ Increment self.global_time variable for animating. """
        self.global_time = self.global_time + 0.1
        #self.paintGL()

    def initializeGL(self):
        """ Initialize OpenGL and related things. """
        func = self.context().functions()
        print(func.glGetString(gl.GL_RENDERER) + " OpenGL " + func.glGetString(gl.GL_VERSION))

        self.shader_vertex_ = QOpenGLShader(QOpenGLShader.Vertex)
        self.shader_vertex_.compileSourceCode(
            "#version 120\n" +
            "attribute vec3 position;\n" +
            "void main()\n" +
            "{\n" +
            "gl_Position = vec4(position, 1.0);\n" +
            "}\n"
        )
        self.shader_fragment_ = QOpenGLShader(QOpenGLShader.Fragment)

        self.shader_template_pre_ = \
            "#version 120\n" \
            "uniform vec3 iResolution;				// viewport resolution (in pixels)\n" \
            "uniform vec2 iGlobalTime;				// shader playback time (in seconds)\n" \
            "\n "
        self.shader_template_post_ = \
            "\n" \
            "void main()\n" \
            "{\n" \
            "mainImage(gl_FragColor, gl_FragCoord.xy);\n" \
            "}\n"

        self.shader_fallback_ = \
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n" \
            "{\n" \
            "vec2 uv = fragCoord.xy / iResolution.xy;\n" \
            "fragColor = vec4(uv,0.5+0.5*sin(iGlobalTime.x),1.0);\n" \
            "}\n"
        self.shader_user_ = self.shader_fallback_

        self.shader_fragment_.compileSourceCode(self.shader_template_pre_ + self.shader_user_ + self.shader_template_post_)

        self.program_ = QOpenGLShaderProgram()
        self.program_.addShader(self.shader_vertex_)
        self.program_.addShader(self.shader_fragment_)
        self.program_.link()

        self.attrib_position = self.program_.attributeLocation("position")
        self.uniform_iGlobalTime = self.program_.uniformLocation("iGlobalTime")
        self.uniform_iResolution = self.program_.uniformLocation("iResolution")

        self.program_.release()

    def set_shader(self, user_shader: str):
        """ Replace part of the fragment shader. """
        self.timer_.stop()
        self.global_time = 0.0

        self.makeCurrent()
        self.shader_user_ = user_shader
        self.shader_fragment_.compileSourceCode(self.shader_template_pre_ + self.shader_user_ + self.shader_template_post_)
        self.program_.removeAllShaders()
        self.program_.addShader(self.shader_vertex_)
        self.program_.addShader(self.shader_fragment_)
        self.program_.link()
        self.program_.bind()

        self.attrib_position = self.program_.attributeLocation("position")
        self.uniform_iGlobalTime = self.program_.uniformLocation("iGlobalTime")
        self.uniform_iResolution = self.program_.uniformLocation("iResolution")

        self.timer_.start()
        self.program_.release()
        self.doneCurrent()

    def get_shader(self) -> str:
        return self.shader_user_

    def resizeGL(self, width, height):
        """ Resize OGL window. """
        self.width_ = width
        self.height_ = height
        func = self.context().functions()
        func.glViewport(0, 0, self.width_, self.height_)
        self.parent().resize(self.width_, self.height_)


    def paintGL(self):
        """ Paint it! """
        func = self.context().functions()
        func.glClearColor(0.0, 0.0, 0.0, 1.0)
        func.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        func.glFrontFace(gl.GL_CW)
        func.glCullFace(gl.GL_FRONT)
        func.glEnable(gl.GL_CULL_FACE)
        func.glEnable(gl.GL_DEPTH_TEST)

        self.program_.bind()
        self.program_.setUniformValue(self.uniform_iGlobalTime, self.global_time, 0.0)
        self.program_.setUniformValue(self.uniform_iResolution, float(self.width_), float(self.height_), 0.0)

        self.program_.setAttributeArray(self.attrib_position, self.vertices_, 2, 0)
        func.glEnableVertexAttribArray(self.attrib_position)
        func.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        func.glDisableVertexAttribArray(self.attrib_position)

        self.program_.release()
        func.glDisable(gl.GL_DEPTH_TEST)
        func.glDisable(gl.GL_CULL_FACE)
        self.update()
