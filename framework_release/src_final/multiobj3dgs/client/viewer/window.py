"""Native GLFW/GL window. All methods run on the presentation owner thread."""
import os
import time


class NativeWindow:
    def __init__(self, controller, runtime, width=1024, height=768):
        if os.environ.get('PYOPENGL_PLATFORM') == 'egl':
            raise RuntimeError('Native display requires the desktop OpenGL platform; unset PYOPENGL_PLATFORM=egl')
        import glfw
        # FAST detects an existing context through PyOpenGL's import state.
        from OpenGL import GL
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed: run native mode on a machine with DISPLAY/Wayland, or use --mode headless')
        self.glfw,self.GL = glfw,GL
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_COMPAT_PROFILE)
        self.window = glfw.create_window(width,height,'Multi-object StreamSTGS DASH',None,None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('An OpenGL 4.3 window could not be created')
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.controller,self.runtime = controller,runtime
        self._cursor = None
        self._last = time.monotonic()
        self.imgui,self.ui = None,None
        self._selected_object = 0
        try:
            import imgui
            from imgui.integrations.glfw import GlfwRenderer
            imgui.create_context()
            self.imgui,self.ui = imgui,GlfwRenderer(self.window,attach_callbacks=False)
        except ImportError:
            pass
        glfw.set_key_callback(self.window,self._key)
        glfw.set_char_callback(self.window,self._char)
        glfw.set_scroll_callback(self.window,self._scroll)
        glfw.set_framebuffer_size_callback(self.window,self._resize)
        self._resize(self.window,*glfw.get_framebuffer_size(self.window))

    def _capture(self):
        return self.imgui and self.imgui.get_io().want_capture_mouse

    def _char(self,window,codepoint):
        if self.ui and hasattr(self.ui,'char_callback'):
            self.ui.char_callback(window,codepoint)

    def _key(self,window,key,scancode,action,mods):
        if self.ui:
            self.ui.keyboard_callback(window,key,scancode,action,mods)
        g = self.glfw
        if action != g.PRESS:
            return
        if self.imgui and self.imgui.get_io().want_capture_keyboard:
            return
        if key == g.KEY_ESCAPE:
            g.set_window_should_close(window,True)
        elif key == g.KEY_SPACE:
            self.runtime.clock.pause() if self.runtime.clock.playing else self.runtime.clock.play()
        elif key in (g.KEY_LEFT,g.KEY_RIGHT):
            self.runtime.seek(self.runtime.clock.position+(-2 if key==g.KEY_LEFT else 2))
        elif key == g.KEY_HOME:
            self.runtime.seek(0)
        elif key == g.KEY_L:
            self.runtime.clock.loop = not self.runtime.clock.loop
        elif key == g.KEY_F:
            self.controller.free = not self.controller.free
        elif key == g.KEY_TAB and self.runtime.catalog.instances:
            self._selected_object = (self._selected_object+1)%len(self.runtime.catalog.instances)
        elif key == g.KEY_Q and self.runtime.catalog.instances:
            instance = self.runtime.catalog.instances[self._selected_object]
            reps = list(self.runtime.catalog.contents[instance.content_id].representations)
            current = self.runtime.selector.selected[instance.id]
            self.runtime.request_quality(instance.id,reps[(reps.index(current)+1)%len(reps)])

    def _scroll(self,window,x,y):
        if self.ui:
            self.ui.scroll_callback(window,x,y)
        if not self._capture():
            self.controller.zoom(y)

    def _resize(self,window,width,height):
        if width>0 and height>0:
            if self.ui and hasattr(self.ui,'resize_callback'):
                self.ui.resize_callback(window,width,height)
            self.controller.resize(width,height)
            self.GL.glViewport(0,0,width,height)

    @property
    def closed(self):
        return self.glfw.window_should_close(self.window)

    def poll(self):
        g=self.glfw
        g.poll_events()
        now=time.monotonic();dt=min(.1,now-self._last);self._last=now
        x,y=g.get_cursor_pos(self.window)
        if self._cursor and not self._capture():
            dx,dy=x-self._cursor[0],y-self._cursor[1]
            if g.get_mouse_button(self.window,g.MOUSE_BUTTON_LEFT)==g.PRESS:
                self.controller.orbit(dx,dy)
            elif g.get_mouse_button(self.window,g.MOUSE_BUTTON_RIGHT)==g.PRESS:
                self.controller.pan(dx,dy)
        self._cursor=(x,y)
        if self.controller.free and not (self.imgui and self.imgui.get_io().want_capture_keyboard):
            pressed=lambda k:int(g.get_key(self.window,k)==g.PRESS)
            right=pressed(g.KEY_D)-pressed(g.KEY_A)
            down=pressed(g.KEY_E)-pressed(g.KEY_C)
            forward=pressed(g.KEY_W)-pressed(g.KEY_S)
            if right or down or forward:
                self.controller.move(right,down,forward,dt)

    def _ui_diagnostics(self):
        """Allow integrations to throttle snapshots without replacing controls."""
        return self.runtime.diagnostics()

    def _draw_extra_ui(self, diagnostic):
        """Optional integration status inside the existing playback panel."""
        pass

    def draw_ui(self):
        diagnostic=self._ui_diagnostics()
        self.glfw.set_window_title(self.window,
            f'StreamSTGS | t={diagnostic["scene_time"]:.2f}s | frames={diagnostic["frames"]} | '
            f'{"buffering" if diagnostic["stalled"] else "playing" if diagnostic["playing"] else "paused"} | '
            f'GPU models {diagnostic["gpu_bytes"]/1024**2:.0f} MiB')
        if not self.ui:
            return
        im=self.imgui
        self.ui.process_inputs();im.new_frame()
        im.begin('Playback and objects')
        if im.button('Pause' if self.runtime.clock.playing else 'Play'):
            self.runtime.clock.pause() if self.runtime.clock.playing else self.runtime.clock.play()
        im.same_line()
        changed,loop=im.checkbox('Loop',self.runtime.clock.loop)
        if changed:self.runtime.clock.loop=loop
        changed,value=im.slider_float('Timeline',self.runtime.clock.position,0.,self.runtime.clock.duration,format='%.2f s')
        if changed:self.runtime.seek(value)
        im.text(f'Fresh wall FPS: {diagnostic["fresh_wall_fps"]:.1f}' if diagnostic['fresh_wall_fps'] is not None else 'Fresh wall FPS: collecting')
        if diagnostic['torch_allocated_bytes'] is not None:
            im.text(f'CUDA allocated {diagnostic["torch_allocated_bytes"]/1024**2:.0f} MiB | reserved {diagnostic["torch_reserved_bytes"]/1024**2:.0f} MiB')
        im.text(f'Rebuffer: {diagnostic["rebuffer_count"]} / {diagnostic["rebuffer_seconds"]:.2f}s')
        im.text(f'Host {diagnostic["host_bytes"]/1024**2:.0f} MiB | GPU models {diagnostic["gpu_bytes"]/1024**2:.0f} MiB')
        im.text('LMB orbit; RMB pan; wheel zoom; F fly; WASD/E/C move')
        for instance in self.runtime.catalog.instances:
            reps=list(self.runtime.catalog.contents[instance.content_id].representations)
            selected=self.runtime.selector.selected[instance.id]
            changed,index=im.combo(instance.id,reps.index(selected),reps)
            if changed:self.runtime.request_quality(instance.id,reps[index])
            for job in diagnostic['objects']:
                if job['instance_id']==instance.id:
                    im.text(f'  {job["gop_id"]}: {job["state"]} ({job["representation_id"]})')
        self._draw_extra_ui(diagnostic)
        im.end();im.render();self.ui.render(im.get_draw_data())

    def swap(self):
        self.glfw.swap_buffers(self.window)

    def close(self):
        if self.ui:self.ui.shutdown()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()
