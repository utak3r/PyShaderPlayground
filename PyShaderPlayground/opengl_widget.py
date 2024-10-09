from OpenGL import GL as gl
from PySide6.QtWidgets import QMessageBox
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat, QOpenGLFunctions, QImage
from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram, QOpenGLFramebufferObject, QOpenGLTexture
from PySide6.QtCore import QTimer
import os
from pathlib import Path
from PyShaderPlayground.ShaderPlaygroundInputs import InputTexture, InputTexture2D, InputTextureSound

class ShaderWidget(QOpenGLWidget, QOpenGLFunctions):
    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)
        QOpenGLFunctions.__init__(self)

        OpenGL_format = QSurfaceFormat.defaultFormat()
        OpenGL_format.setProfile(QSurfaceFormat.defaultFormat().profile())
        #OpenGL_format.setVersion(3, 1)
        OpenGL_format.setDepthBufferSize(24)
        OpenGL_format.setStencilBufferSize(8)
        QSurfaceFormat.setDefaultFormat(OpenGL_format)

        self.width_ = self.width()
        self.height_ = self.height()
        self.vao_ = None
        self.program_ = None
        self.shader_vertex_ = None
        self.shader_fragment_ = None
        self.attrib_position = None
        self.uniform_iResolution = None
        self.uniform_iMouse = None
        self.uniform_iGlobalTime = None
        self.uniform_iChannel0 = None
        self.uniform_iChannel1 = None
        self.texture_0_ = InputTexture()
        self.texture_1_ = InputTexture()
        self.global_time: float = 0.0
        self.mouse = [0.0, 0.0, 0.0, 0.0]
        self.framerate_ = 50
        self.anim_speed_ = 1.0
        self.anim_speed_modifier_ = 1.0
        self.shader_template_pre_ = ""
        self.shader_template_post_ = ""
        self.shader_user_ = \
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )\n" \
            "{\n" \
            "// You can use:\n" \
            "// vec3 iResolution - The viewport resolution (z is pixel aspect ratio, usually 1.0)\n" \
            "// float iTime - shader playback time (in seconds)\n" \
            "// vec4 iMouse - mouse pixel coords. xy: current (if MLB down), zw: click. Range [-1.0 - 1.0], [0,0] in the middle\n" \
            "// sampler2D iChannel0 - Sampler for input texture\n\n" \
            "vec2 uv = fragCoord.xy / iResolution.xy;\n" \
            "fragColor = vec4(uv,0.5+0.5*sin(iTime),1.0);\n" \
            "}\n"
        self.shader_fallback_ = ""
        
        self.vertices_ = [ -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ]
        self.resize(self.width_, self.height_)

        self.timer_ = QTimer(self)
        self.timer_.timeout.connect(self.timer_tick)
        self.set_animation_speed(1.0, 30)
        self.timer_.start()

    def set_animation_speed(self, speed: float=1.0, framerate: float=50):
        """ How many "ones" per second? And what's a desired framerate? """
        self.framerate_ = framerate
        self.anim_speed_ = speed
        self.timer_.setInterval(1000 / self.framerate_)

    def timer_tick(self):
        """ Increment self.global_time variable for animating. """
        self.global_time = self.global_time + (self.anim_speed_modifier_ * self.anim_speed_ * (1.0 / self.framerate_))

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
            self.global_time = self.global_time + (self.anim_speed_modifier_ * self.anim_speed_ * (1.0 / self.framerate_))

    def increment_animation(self, frames: int):
        """ Advance animation for given frames number. """
        self.global_time = self.global_time + frames * (self.anim_speed_ * (1.0 / self.framerate_))

    def initializeGL(self):
        """ Initialize OpenGL and related things. """
        self.initializeOpenGLFunctions()
        #func = self.context().functions()
        #print(func.glGetString(gl.GL_RENDERER) + " OpenGL " + func.glGetString(gl.GL_VERSION))

        self.shader_vertex_ = QOpenGLShader(QOpenGLShader.Vertex)
        self.shader_vertex_.compileSourceCode(
            "#version 150\n" +
            "attribute vec3 position;\n" +
            "void main()\n" +
            "{\n" +
            "gl_Position = vec4(position, 1.0);\n" +
            "}\n"
        )
        self.shader_fragment_ = QOpenGLShader(QOpenGLShader.Fragment)

        self.shader_template_pre_ = \
            "#version 150\n" \
            "uniform vec3 iResolution;				// The viewport resolution (z is pixel aspect ratio, usually 1.0)\n" \
            "uniform vec2 iGlobalTime;				// shader playback time (in seconds)\n" \
            "uniform vec4 iMouse;                   // mouse pixel coords. xy: current (if MLB down), zw: click\n" \
            "uniform sampler2D iChannel0;			// Sampler for input texture\n" \
            "uniform sampler2D iChannel1;			// Sampler for input texture\n" \
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
        self.uniform_iMouse = self.program_.uniformLocation("iMouse")
        self.uniform_iChannel0 = self.program_.uniformLocation("iChannel0")
        self.uniform_iChannel1 = self.program_.uniformLocation("iChannel1")

        self.program_.release()

    def set_shader(self, user_shader: str):
        """ Replace part of the fragment shader. """
        self.timer_.stop()
        self.global_time = 0.0

        self.makeCurrent()
        self.shader_user_ = user_shader
        if not self.shader_fragment_.compileSourceCode(self.shader_template_pre_ + self.shader_user_ + self.shader_template_post_):
            log = self.shader_fragment_.log()
            # for debug:
            with open('fragment_shader.temp.glsl', 'w') as f:
                f.write(self.shader_fragment_.sourceCode().toStdString())
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
            self.uniform_iMouse = self.program_.uniformLocation("iMouse")
            self.uniform_iChannel0 = self.program_.uniformLocation("iChannel0")
            self.uniform_iChannel1 = self.program_.uniformLocation("iChannel1")

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
        #func = self.context().functions()
        self.glViewport(0, 0, self.width_, self.height_)


    def paintGL(self):
        """ Paint it! """
        #func = self.context().functions()
        self.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.glFrontFace(gl.GL_CW)
        self.glCullFace(gl.GL_FRONT)
        self.glEnable(gl.GL_CULL_FACE)
        self.glEnable(gl.GL_DEPTH_TEST)

        self.program_.bind()
        self.program_.setUniformValue(self.uniform_iGlobalTime, self.global_time, 0.0)
        self.program_.setUniformValue(self.uniform_iResolution, float(self.width_), float(self.height_), 0.0)
        self.program_.setUniformValue(self.uniform_iMouse, self.mouse[0], self.mouse[1], self.mouse[2], self.mouse[3])        
        self.texture_0_.set_position(self.global_time)
        if self.texture_0_.can_be_binded():
            self.texture_0_.bind()
        self.program_.setUniformValue(self.uniform_iChannel0, int(0))
        self.texture_1_.set_position(self.global_time)
        if self.texture_1_.can_be_binded():
            self.texture_1_.bind()
        self.program_.setUniformValue(self.uniform_iChannel1, int(1))

        self.program_.setAttributeArray(self.attrib_position, self.vertices_, 2, 0)
        self.glEnableVertexAttribArray(self.attrib_position)
        self.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        self.glDisableVertexAttribArray(self.attrib_position)

        self.program_.release()
        self.glDisable(gl.GL_DEPTH_TEST)
        self.glDisable(gl.GL_CULL_FACE)
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
        # Ok, here's the thing...
        # Without below current screen buffer grab...
        # following FBO doesn't work as expected??!
        # WTF...
        fbImage = self.grabFramebuffer()
        #fbImage.save(f'{name}_screenbuffer.{ext}', img_type, 95)
        # create an offscreen frame buffer
        buffer = QOpenGLFramebufferObject(width, height)
        if buffer.bind():
            self.resizeGL(width, height)
            self.paintGL()
            # save image
            fboImage = buffer.toImage()
            #fboImage.save(f'{name}_unpremultiplied.{ext}', img_type, 95)
            # deal with unpremultiplied image
            image = QImage(fboImage.constBits(), fboImage.width(), fboImage.height(), QImage.Format.Format_ARGB32)
            image.save(filename, img_type, 95)
            # restore screen rendering
            buffer.release()
            self.resizeGL(orig_width, orig_height)
        else:
            print("ShaderWidget.render_image: Unable to switch rendering from screen to FBO.")


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
                file_ext = Path(image).suffix
                if file_ext.casefold() == ".jpg" or file_ext.casefold() == ".png":
                    self.texture_0_ = InputTexture2D(image)
                elif file_ext.casefold() == ".wav" or file_ext.casefold() == ".mp3":
                    self.texture_0_ = InputTextureSound(image)
        elif channel == 1:
            if self.isValid():
                file_ext = Path(image).suffix
                if file_ext.casefold() == ".jpg" or file_ext.casefold() == ".png":
                    self.texture_1_ = InputTexture2D(image)
                elif file_ext.casefold() == ".wav" or file_ext.casefold() == ".mp3":
                    self.texture_1_ = InputTextureSound(image)

    def get_texture(self, channel: int):
        if channel == 0:
            if self.isValid():
                return self.texture_0_
        elif channel == 1:
            if self.isValid():
                return self.texture_1_

    def get_texture_thumbnail(self, channel: int):
        if channel == 0:
            if self.texture_0_ is not None:
                return self.texture_0_.get_thumbnail()
        elif channel == 1:
            if self.texture_1_ is not None:
                return self.texture_1_.get_thumbnail()

    @staticmethod
    def make_texture(image: str) -> QOpenGLTexture:
        """ Create texture from given filename. """
        texture = QOpenGLTexture(QOpenGLTexture.Target2D)
        texture.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        texture.setMagnificationFilter(QOpenGLTexture.Linear)
        texture.setWrapMode(QOpenGLTexture.Repeat)
        texture.setData(QImage(image).mirrored())
        return texture

    def mousePressEvent(self, event):
        x = event.localPos().x()
        y = event.localPos().y()
        self.mouse[0] = float(x) / float(self.width_) * 2.0 - 1.0
        self.mouse[1] = float(y) / float(self.height_) * 2.0 - 1.0
        self.mouse[2] = self.mouse[0]
        self.mouse[3] = self.mouse[1]
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        x = event.localPos().x()
        y = event.localPos().y()
        self.mouse[0] = float(x) / float(self.width_) * 2.0 - 1.0
        self.mouse[1] = float(y) / float(self.height_) * 2.0 - 1.0
        super().mouseMoveEvent(event)
