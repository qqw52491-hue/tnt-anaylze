//! Main-thread lifecycle for the HUD's own HighGUI window.
//! Cocoa HighGUI creates a window without NSWindowStyleMaskClosable. Enable it
//! on this process's named window only; never message another app or process.
use opencv::highgui;

pub struct HudWindow {
    name: String,
    #[cfg(target_os = "macos")]
    native: Option<cocoa::Window>,
}

impl HudWindow {
    /// Attach only after the first imshow, so a visible window already exists.
    pub fn attach(name: &str) -> Self {
        #[cfg(target_os = "macos")]
        let native = cocoa::Window::attach(name);
        Self {
            name: name.to_owned(),
            #[cfg(target_os = "macos")]
            native,
        }
    }

    pub fn is_closed(&self) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(window) = &self.native {
            return window.is_closed();
        }
        // A destroyed window can raise instead of returning zero. Do not turn
        // that error into "visible", then let imshow recreate the window.
        closed_property(highgui::get_window_property(&self.name, highgui::WND_PROP_VISIBLE).ok())
    }
}

fn closed_property(value: Option<f64>) -> bool {
    matches!(value, None | Some(0.0))
}

#[cfg(target_os = "macos")]
mod cocoa {
    use std::ffi::{c_char, c_void, CStr};
    type Id = *mut c_void;
    type Sel = *mut c_void;
    const CLOSABLE: usize = 1 << 1;

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
    }
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    // Typed message calls keep pointer, NSUInteger and BOOL return ABIs distinct.
    unsafe fn object(receiver: Id, selector: &CStr) -> Id {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel) -> Id = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()))
        }
    }
    unsafe fn integer(receiver: Id, selector: &CStr) -> usize {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel) -> usize = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()))
        }
    }
    unsafe fn void(receiver: Id, selector: &CStr) {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel) = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()));
        }
    }
    unsafe fn boolean(receiver: Id, selector: &CStr) -> bool {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel) -> bool = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()))
        }
    }
    unsafe fn indexed(receiver: Id, selector: &CStr, index: usize) -> Id {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel, usize) -> Id = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()), index)
        }
    }
    unsafe fn set_integer(receiver: Id, selector: &CStr, value: usize) {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel, usize) = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()), value);
        }
    }
    unsafe fn action(receiver: Id, selector: &CStr) {
        unsafe {
            let send: unsafe extern "C" fn(Id, Sel, Id) = std::mem::transmute(objc_msgSend as *const ());
            send(receiver, sel_registerName(selector.as_ptr()), std::ptr::null_mut());
        }
    }
    struct Pool(Id);
    impl Drop for Pool {
        fn drop(&mut self) { unsafe { void(self.0, c"drain"); } }
    }

    pub(super) struct Window {
        window: Id,
        app: Id,
        original_mask: usize,
    }
    impl Window {
        pub(super) fn attach(name: &str) -> Option<Self> {
            unsafe {
                if !boolean(objc_getClass(c"NSThread".as_ptr()), c"isMainThread") { return None; }
                let _pool = Pool(object(objc_getClass(c"NSAutoreleasePool".as_ptr()), c"new"));
                let app = object(objc_getClass(c"NSApplication".as_ptr()), c"sharedApplication");
                let windows = object(app, c"windows");
                for i in 0..integer(windows, c"count").min(1000) {
                    let window = indexed(windows, c"objectAtIndex:", i);
                    let title = object(object(window, c"title"), c"UTF8String") as *const c_char;
                    if title.is_null() || CStr::from_ptr(title).to_bytes() != name.as_bytes() { continue; }
                    let mask = integer(window, c"styleMask");
                    set_integer(window, c"setStyleMask:", mask | CLOSABLE);
                    object(window, c"retain");
                    return Some(Self { window, app, original_mask: mask });
                }
            }
            None
        }
        pub(super) fn is_closed(&self) -> bool {
            unsafe {
                !boolean(self.window, c"isVisible")
                    && !boolean(self.window, c"isMiniaturized")
                    && !boolean(self.app, c"isHidden")
            }
        }
        pub(super) fn close_enabled(&self) -> bool {
            unsafe {
                let button = indexed(self.window, c"standardWindowButton:", 0);
                !button.is_null() && boolean(button, c"isEnabled")
            }
        }
        pub(super) fn miniaturize(&self) { unsafe { action(self.window, c"miniaturize:"); } }
        pub(super) fn restore(&self) { unsafe { action(self.window, c"deminiaturize:"); } }
        pub(super) fn click_close(&self) {
            unsafe { action(indexed(self.window, c"standardWindowButton:", 0), c"performClick:"); }
        }
        pub(super) fn original_closable(&self) -> bool { self.original_mask & CLOSABLE != 0 }
    }
    impl Drop for Window {
        fn drop(&mut self) { unsafe { void(self.window, c"release"); } }
    }
}

/// Explicit, isolated diagnostic. Opens only its own small test window; never
/// starts recorders, touches the game, or closes an existing HUD window.
#[cfg(target_os = "macos")]
pub fn check_native_close() -> opencv::Result<()> {
    use opencv::{core, imgproc};
    let name = format!("TNT isolated close check {}", std::process::id());
    highgui::named_window(&name, highgui::WINDOW_AUTOSIZE)?;
    let mut image = core::Mat::new_rows_cols_with_default(100, 340, core::CV_8UC3, core::Scalar::all(30.0))?;
    imgproc::put_text(&mut image, "Isolated HUD close check", core::Point::new(15,55),
        imgproc::FONT_HERSHEY_SIMPLEX, 0.6, core::Scalar::all(220.0), 1, imgproc::LINE_AA, false)?;
    highgui::imshow(&name, &image)?;
    highgui::wait_key(30)?;
    let guard = HudWindow::attach(&name);
    let checked = (|| {
        let window = guard.native.as_ref().ok_or_else(|| opencv::Error::new(core::StsError, "native test window not found"))?;
        println!("original_closable={} close_button_enabled={}", window.original_closable(), window.close_enabled());
        if !window.close_enabled() { return Err(opencv::Error::new(core::StsError, "close button disabled")); }
        window.miniaturize();
        highgui::wait_key(100)?;
        if guard.is_closed() { return Err(opencv::Error::new(core::StsError, "minimize mistaken for close")); }
        window.restore();
        highgui::wait_key(100)?;
        window.click_close();
        highgui::wait_key(50)?;
        if !guard.is_closed() { return Err(opencv::Error::new(core::StsError, "native close not detected")); }
        println!("minimize_keeps_running=true native_red_button_close_detected=true");
        Ok(())
    })();
    let _ = highgui::destroy_window(&name);
    checked
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_window_is_closed_and_unsupported_property_is_not() {
        assert!(closed_property(None));
        assert!(closed_property(Some(0.0)));
        assert!(!closed_property(Some(1.0)));
        assert!(!closed_property(Some(-1.0)));
    }
}
