"""
🎨 THEME CODE GENERATOR 🌈
Generates beautiful theme management code! ✨
"""

class ThemeCodeGenerator:
    """Generates fabulous theme management code! 🎨💖"""
    
    def __init__(self):
        self.include_advanced_features = True
        
    def generate_theme_class(self):
        """Generate the complete FabulousTheme class! 🎨"""
        theme_code = '''class FabulousTheme:
    """Fabulous theme management! 🎨✨"""
    
    def __init__(self, theme_data):
        self.theme = theme_data
        
    def apply_to_widget(self, widget, widget_type="default"):
        """Apply theme colors to any widget! 🌈"""
        try:
            if widget_type == "button":
                self._apply_button_theme(widget)
            elif widget_type == "entry":
                self._apply_entry_theme(widget)
            elif widget_type == "text":
                self._apply_text_theme(widget)
            elif widget_type == "label":
                self._apply_label_theme(widget)
            elif widget_type == "frame":
                self._apply_frame_theme(widget)
            else:
                self._apply_default_theme(widget)
        except Exception as e:
            # Silently handle theming errors - some widgets don't support all options
            pass
            
    def _apply_button_theme(self, widget):
        """Apply button-specific theming! 🔘"""
        widget.configure(
            bg=self.theme["widget"],
            fg=self.theme["text"],
            activebackground=self.theme["accent"],
            activeforeground=self.theme["text"],
            relief="raised",
            font=("Arial", 10, "bold"),
            borderwidth=2
        )
        
    def _apply_entry_theme(self, widget):
        """Apply entry-specific theming! 📝"""
        widget.configure(
            bg=self.theme["widget"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            selectbackground=self.theme["accent"],
            selectforeground=self.theme["window"],
            relief="sunken",
            font=("Arial", 10),
            borderwidth=1
        )
        
    def _apply_text_theme(self, widget):
        """Apply text widget theming! 📄"""
        widget.configure(
            bg=self.theme["widget"],
            fg=self.theme["text"],
            insertbackground=self.theme["text"],
            selectbackground=self.theme["accent"],
            selectforeground=self.theme["window"],
            relief="sunken",
            font=("Arial", 10),
            borderwidth=1,
            wrap="word"
        )
        
    def _apply_label_theme(self, widget):
        """Apply label theming! 📝"""
        widget.configure(
            bg=self.theme["widget"],
            fg=self.theme["text"],
            font=("Arial", 10)
        )
        
    def _apply_frame_theme(self, widget):
        """Apply frame theming! 📦"""
        widget.configure(
            bg=self.theme["widget"],
            relief="groove",
            borderwidth=1
        )
        
    def _apply_default_theme(self, widget):
        """Apply default theming to any widget! 🌈"""
        try:
            widget.configure(
                bg=self.theme["widget"],
                fg=self.theme["text"]
            )
        except:
            # Some widgets don't support bg/fg
            pass
            
    def get_color(self, color_type):
        """Get a theme color safely! 🎨"""
        color_map = {
            'background': 'window',
            'foreground': 'text',
            'highlight': 'accent',
            'border': 'border'
        }
        
        key = color_map.get(color_type, color_type)
        return self.theme.get(key, "#000000")
        
    def create_gradient_color(self, base_color, steps=10):
        """Create gradient colors for advanced effects! ✨"""
        # Simple gradient generation - can be enhanced
        colors = []
        for i in range(steps):
            # This is a simplified version - could use proper color mixing
            colors.append(base_color)
        return colors
        
    def apply_hover_effects(self, widget, widget_type="button"):
        """Add hover effects to widgets! ✨"""
        if widget_type == "button":
            def on_enter(e):
                widget.configure(bg=self.lighten_color(self.theme["widget"]))
            def on_leave(e):
                widget.configure(bg=self.theme["widget"])
                
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            
    def lighten_color(self, color, factor=1.2):
        """Lighten a color for hover effects! 💡"""
        # Simple color lightening - could be more sophisticated
        if color.startswith('#'):
            try:
                # Convert hex to RGB, lighten, convert back
                hex_color = color[1:]
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                lightened = tuple(min(255, int(c * factor)) for c in rgb)
                return f"#{lightened[0]:02x}{lightened[1]:02x}{lightened[2]:02x}"
            except:
                return color
        return color
        
    def darken_color(self, color, factor=0.8):
        """Darken a color for pressed effects! 🌑"""
        if color.startswith('#'):
            try:
                hex_color = color[1:]
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                darkened = tuple(max(0, int(c * factor)) for c in rgb)
                return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"
            except:
                return color
        return color
        
    def get_contrasting_color(self, background_color):
        """Get a contrasting text color for readability! 👁️"""
        # Simple contrast calculation - could be more sophisticated
        if background_color.startswith('#'):
            try:
                hex_color = background_color[1:]
                rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                return "#000000" if brightness > 128 else "#ffffff"
            except:
                return "#000000"
        return "#000000"
        
    def apply_theme_to_all_children(self, parent_widget):
        """Recursively apply theme to all child widgets! 🌳"""
        for child in parent_widget.winfo_children():
            # Determine widget type and apply appropriate theme
            widget_class = child.winfo_class()
            widget_type = widget_class.lower()
            
            if widget_type in ['button']:
                self.apply_to_widget(child, 'button')
            elif widget_type in ['entry']:
                self.apply_to_widget(child, 'entry')
            elif widget_type in ['text']:
                self.apply_to_widget(child, 'text')
            elif widget_type in ['label']:
                self.apply_to_widget(child, 'label')
            elif widget_type in ['frame', 'labelframe']:
                self.apply_to_widget(child, 'frame')
            else:
                self.apply_to_widget(child)
                
            # Recursively apply to children
            self.apply_theme_to_all_children(child)

'''
        return theme_code
        
    def generate_theme_utilities(self):
        """Generate additional theme utility functions! 🛠️"""
        utilities = '''
# 🎨 Theme Utility Functions! ✨

def create_custom_theme(name, window_color, widget_color, text_color, border_color, accent_color):
    """Create a custom theme! 🌈"""
    return {
        'name': name,
        'window': window_color,
        'widget': widget_color,
        'text': text_color,
        'border': border_color,
        'accent': accent_color
    }

def blend_colors(color1, color2, ratio=0.5):
    """Blend two colors together! 🎨"""
    if not (color1.startswith('#') and color2.startswith('#')):
        return color1
        
    try:
        # Convert hex to RGB
        rgb1 = tuple(int(color1[1:][i:i+2], 16) for i in (0, 2, 4))
        rgb2 = tuple(int(color2[1:][i:i+2], 16) for i in (0, 2, 4))
        
        # Blend
        blended = tuple(int(c1 * (1-ratio) + c2 * ratio) for c1, c2 in zip(rgb1, rgb2))
        
        return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"
    except:
        return color1

def get_theme_preview_colors(theme_data):
    """Get a preview palette of theme colors! 🎨"""
    return [
        theme_data.get('window', '#ffffff'),
        theme_data.get('widget', '#f0f0f0'),
        theme_data.get('text', '#000000'),
        theme_data.get('border', '#cccccc'),
        theme_data.get('accent', '#0078d4')
    ]
'''
        return utilities
        
    def generate_animation_support(self):
        """Generate animation support for themes! ✨"""
        animation_code = '''
# ✨ Theme Animation Support! 🌈

class ThemeAnimator:
    """Animate theme transitions! ✨"""
    
    def __init__(self, theme_manager):
        self.theme_manager = theme_manager
        self.animation_speed = 50  # milliseconds
        self.animation_steps = 10
        
    def animate_theme_change(self, widget, old_theme, new_theme, callback=None):
        """Animate transition between themes! 🎭"""
        steps = self.animation_steps
        current_step = 0
        
        def animate_step():
            nonlocal current_step
            if current_step >= steps:
                # Animation complete
                self.theme_manager.apply_to_widget(widget)
                if callback:
                    callback()
                return
                    
            # Calculate intermediate colors
            ratio = current_step / steps
            
            # This is a simplified version - could interpolate all theme colors
            intermediate_bg = self.interpolate_color(
                old_theme.get('widget', '#ffffff'),
                new_theme.get('widget', '#ffffff'),
                ratio
            )
            
            try:
                widget.configure(bg=intermediate_bg)
            except:
                pass
                
            current_step += 1
            widget.after(self.animation_speed, animate_step)
            
        animate_step()
        
    def interpolate_color(self, color1, color2, ratio):
        """Interpolate between two colors! 🌈"""
        if not (color1.startswith('#') and color2.startswith('#')):
            return color2
            
        try:
            rgb1 = tuple(int(color1[1:][i:i+2], 16) for i in (0, 2, 4))
            rgb2 = tuple(int(color2[1:][i:i+2], 16) for i in (0, 2, 4))
            
            interpolated = tuple(int(c1 * (1-ratio) + c2 * ratio) for c1, c2 in zip(rgb1, rgb2))
            
            return f"#{interpolated[0]:02x}{interpolated[1]:02x}{interpolated[2]:02x}"
        except:
            return color2
            
    def pulse_widget(self, widget, color, duration=1000):
        """Make a widget pulse with color! 💓"""
        original_bg = widget.cget('bg')
        steps = 20
        step_duration = duration // (steps * 2)
        
        def pulse_step(step, direction):
            if step > steps:
                widget.configure(bg=original_bg)
                return
                
            ratio = step / steps if direction == 'up' else (steps - step) / steps
            pulse_color = self.interpolate_color(original_bg, color, ratio * 0.3)
            
            try:
                widget.configure(bg=pulse_color)
            except:
                pass
                
            next_step = step + 1 if direction == 'up' else step - 1
            next_direction = 'down' if direction == 'up' and step >= steps else 'up' if direction == 'down' and step <= 0 else direction
            
            if not (direction == 'down' and step <= 0):
                widget.after(step_duration, lambda: pulse_step(next_step, next_direction))
            else:
                widget.configure(bg=original_bg)
                
        pulse_step(0, 'up')
'''
        return animation_code
        
    def generate_complete_theme_system(self):
        """Generate the complete theme system! 🎨✨"""
        return f"""{self.generate_theme_class()}

{self.generate_theme_utilities()}

{self.generate_animation_support() if self.include_advanced_features else ""}
"""


# 🧪 Test the theme generator
if __name__ == "__main__":
    print("🎨 Testing Theme Code Generator! ✨")
    generator = ThemeCodeGenerator()
    
    theme_code = generator.generate_complete_theme_system()
    print(f"📄 Generated {len(theme_code)} characters of theme code!")
    print("🌈 Theme system generation complete! ✨")