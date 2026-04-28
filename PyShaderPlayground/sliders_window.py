from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QGroupBox, QScrollArea
from PySide6.QtCore import Qt, Signal

class ShaderSlidersWindow(QWidget):
    value_changed = Signal(str, int, float)  # name, index, value

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Shader Parameters")
        self.resize(300, 400)
        
        self.main_layout = QVBoxLayout(self)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_content)
        
        self.main_layout.addWidget(self.scroll)
        
    def update_sliders(self, dynamic_uniforms):
        """ Rebuild the UI based on the new dynamic uniforms. """
        # Clear existing sliders
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        if not dynamic_uniforms:
            self.hide()
            return

        for name, data in dynamic_uniforms.items():
            group = QGroupBox(name)
            group_layout = QVBoxLayout(group)
            
            var_type = data['type']
            s_min = data['min']
            s_max = data['max']
            s_step = data['step']
            current_values = data['value']
            
            num_components = 1
            if var_type.startswith('vec'):
                num_components = int(var_type[-1])
            
            for i in range(num_components):
                comp_layout = QHBoxLayout()
                
                # Component label (X, Y, Z, W)
                comp_labels = ['X', 'Y', 'Z', 'W']
                if num_components > 1:
                    lbl_comp = QLabel(comp_labels[i])
                    lbl_comp.setFixedWidth(15)
                    comp_layout.addWidget(lbl_comp)
                
                # Slider
                slider = QSlider(Qt.Horizontal)
                steps = int((s_max - s_min) / s_step)
                slider.setRange(0, steps)
                
                # Set initial position
                initial_val = current_values[i]
                pos = int((initial_val - s_min) / s_step)
                slider.setValue(pos)
                
                # Value label
                lbl_val = QLabel(f"{initial_val:.2f}")
                lbl_val.setFixedWidth(50)
                
                # Connect slider
                slider.valueChanged.connect(self._make_slider_callback(name, i, s_min, s_step, lbl_val))
                
                comp_layout.addWidget(slider)
                comp_layout.addWidget(lbl_val)
                group_layout.addLayout(comp_layout)
                
            self.scroll_layout.addWidget(group)
            
        self.show()

    def _make_slider_callback(self, name, index, s_min, s_step, lbl_val):
        def callback(pos):
            val = s_min + pos * s_step
            lbl_val.setText(f"{val:.2f}")
            self.value_changed.emit(name, index, val)
        return callback
