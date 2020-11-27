from PySide2.QtWidgets import QOpenGLWidget, QMessageBox
from PySide2.QtGui import QOpenGLShader, QOpenGLShaderProgram, QSurfaceFormat, QOpenGLFramebufferObject, QImage, QOpenGLTexture
from OpenGL import GL as gl
from PySide2.QtCore import QTimer
import os

class ShaderWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Setting up modern OpenGL format
        OpenGL_format = QSurfaceFormat()
        # OpenGL_format.setDepthBufferSize(24)
        # OpenGL_format.setStencilBufferSize(8)
        OpenGL_format.setVersion(1, 2)
        OpenGL_format.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(OpenGL_format)

        self.width_ = self.width()
        self.height_ = self.height()
        self.vao_ = None
        self.program_ = None
        self.shader_vertex_ = None
        self.shader_fragment_ = None
        self.attrib_position = None
        self.uniform_iResolution = None
        self.uniform_iGlobalTime = None
        self.uniform_iChannel0 = None
        self.texture_0_ = None
        self.global_time: float = 0.0
        self.framerate_ = 50
        self.anim_speed_ = 2.0
        self.anim_speed_modifier_ = 1.0
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
        self.set_animation_speed(2.0, 50)
        self.timer_.start()

    def set_animation_speed(self, speed: float=1.0, framerate: float=50):
        """ How many "ones" per second? And what's a desired framerate? """
        self.framerate_ = framerate
        self.anim_speed_ = speed
        self.timer_.setInterval(1000 / self.framerate_)

    def timer_tick(self):
        """ Increment self.global_time variable for animating. """
        self.global_time = self.global_time + (self.anim_speed_modifier_ * self.anim_speed_ * self.framerate_ / 1000.0)

    def animation_framerate(self):
        """ Returns current animation's framerate. """
        return self.framerate_

    def is_playing(self):
        """ Returns True if animation is playing, False otherwise. """
        return self.timer_.isActive()

    def animation_play_pause(self):
        """ Switch between playing/paused states. """
        if self.is_playing():
            self.animation_pause()
        else:
            self.animation_play()

    def animation_pause(self):
        """ Pause animation. """
        self.timer_.stop()

    def animation_play(self):
        """ Play animation. """
        self.timer_.start()
    
    def animation_stop(self):
        """ Stop and rewind. """
        self.animation_pause()
        self.animation_rewind()
    
    def animation_rewind(self):
        """ Rewinds the animation by resetting the global timer counter. """
        self.global_time = 0.0

    def set_animation_speed_modifier(self, value):
        """ Temporary animation speed changing. """
        self.anim_speed_modifier_ = value
        if value != 1.0 and not self.is_playing():
            self.global_time = self.global_time + (self.anim_speed_modifier_ * self.anim_speed_ * self.framerate_ / 1000.0)

    def increment_animation(self, frames: int):
        """ Advance animation for given frames number. """
        self.global_time = self.global_time + frames * (self.anim_speed_ * self.framerate_ / 1000.0)

    def initializeGL(self):
        """ Initialize OpenGL and related things. """
        func = self.context().functions()
        print(func.glGetString(gl.GL_RENDERER) + " OpenGL " + func.glGetString(gl.GL_VERSION))

        self.shader_vertex_ = QOpenGLShader(QOpenGLShader.Vertex)
        self.shader_vertex_.compileSourceCode(
            "#version 130\n" +
            "attribute vec3 position;\n" +
            "void main()\n" +
            "{\n" +
            "gl_Position = vec4(position, 1.0);\n" +
            "}\n"
        )
        self.shader_fragment_ = QOpenGLShader(QOpenGLShader.Fragment)

        self.shader_template_pre_ = \
            "#version 130\n" \
            "uniform vec3 iResolution;				// The viewport resolution (z is pixel aspect ratio, usually 1.0)\n" \
            "uniform vec2 iGlobalTime;				// shader playback time (in seconds)\n" \
            "uniform sampler2D iChannel0;			// Sampler for input texture\n" \
            "float iTime = iGlobalTime.x;\n" \
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
        self.uniform_iChannel0 = self.program_.uniformLocation("iChannel0")

        self.program_.release()

    def set_shader(self, user_shader: str):
        """ Replace part of the fragment shader. """
        self.timer_.stop()
        self.global_time = 0.0

        self.makeCurrent()
        self.shader_user_ = user_shader
        if not self.shader_fragment_.compileSourceCode(self.shader_template_pre_ + self.shader_user_ + self.shader_template_post_):
            log = self.shader_fragment_.log()
            QMessageBox.critical(self, "Shader compile problem", log, QMessageBox.Ok)
        else:
            self.program_.removeAllShaders()
            self.program_.addShader(self.shader_vertex_)
            self.program_.addShader(self.shader_fragment_)
            self.program_.link()
            self.program_.bind()

            self.attrib_position = self.program_.attributeLocation("position")
            self.uniform_iGlobalTime = self.program_.uniformLocation("iGlobalTime")
            self.uniform_iResolution = self.program_.uniformLocation("iResolution")
            self.uniform_iChannel0 = self.program_.uniformLocation("iChannel0")

        self.timer_.start()
        self.program_.release()
        self.doneCurrent()


    def get_shader(self) -> str:
        """ Returns user's part of a shader. """
        return self.shader_user_


    def resizeGL(self, width, height):
        """ Resize OGL window. """
        self.width_ = width
        self.height_ = height
        func = self.context().functions()
        func.glViewport(0, 0, self.width_, self.height_)


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
        if self.texture_0_ is not None:
            self.texture_0_.bind()
        self.program_.setUniformValue(self.uniform_iChannel0, int(0))

        self.program_.setAttributeArray(self.attrib_position, self.vertices_, 2, 0)
        func.glEnableVertexAttribArray(self.attrib_position)
        func.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        func.glDisableVertexAttribArray(self.attrib_position)

        self.program_.release()
        func.glDisable(gl.GL_DEPTH_TEST)
        func.glDisable(gl.GL_CULL_FACE)
        self.update()


    def render_image(self, filename: str, width: int, height: int):
        """ Offscreen rendering with a specified size, saved to a file. """
        name, ext = os.path.splitext(filename)
        img_type = "PNG"
        if ext.casefold() == ".jpg" or ext.casefold() == ".jpeg":
            img_type = "JPG"
        # save screen rendering size
        orig_width = self.width_
        orig_height = self.height_
        # create an offscreen frame buffer
        buffer = QOpenGLFramebufferObject(width, height)
        buffer.bind()
        self.resizeGL(width, height)
        self.paintGL()
        # save image
        image = buffer.toImage()
        image.save(filename, img_type, 90)
        # restore screen rendering
        buffer.release()
        self.resizeGL(orig_width, orig_height)


    @staticmethod
    def maintain_aspect_ratio(width: int, height: int, aspect: float):
        """ Maintain aspect ratio of an image.
        It always modifies the shorter side.
        """
        ret_width = width
        ret_height = height
        if width > height:
            ret_height = width / aspect
        else:
            ret_width = height * aspect
        return [ret_width, ret_height]


    def set_texture(self, channel: int, image: str):
        """ Set texture nr 0 from given filename. """
        if channel == 0:
            if self.isValid():
                self.texture_0_ = ShaderWidget.make_texture(image)

    @staticmethod
    def make_texture(image: str) -> QOpenGLTexture:
        """ Create texture from given filename. """
        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        texture.setMagnificationFilter(QOpenGLTexture.Linear)
        texture.setWrapMode(QOpenGLTexture.Repeat)
        texture.setData(QImage(image).mirrored())
        return texture
