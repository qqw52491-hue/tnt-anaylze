use opencv::{core, highgui, prelude::*};
use std::sync::{Arc, Mutex};

fn main() -> opencv::Result<()> {
    highgui::named_window("Test", highgui::WINDOW_AUTOSIZE)?;
    let state = Arc::new(Mutex::new(0));
    let state_clone = state.clone();
    
    highgui::set_mouse_callback(
        "Test",
        Some(Box::new(move |event, x, y, _flags| {
            if event == highgui::EVENT_LBUTTONDOWN {
                let mut s = state_clone.lock().unwrap();
                *s += 1;
                println!("Clicked at {}, {} - State: {}", x, y, s);
            }
        })),
    )?;
    
    Ok(())
}
