fn main() {
    unsafe {
        let name = std::ffi::CString::new("/usr/lib/libclang.so").unwrap();
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_NOW);
        if handle.is_null() {
            let err = libc::dlerror();
            let err_str = std::ffi::CStr::from_ptr(err).to_string_lossy();
            println!("Failed to load: {}", err_str);
        } else {
            println!("Loaded successfully!");
            libc::dlclose(handle);
        }
    }
}
