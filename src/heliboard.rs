//! Direct JNI bridge used when the Parakeet engine is bundled inside HeliBoard.
//!
//! HeliBoard owns AudioRecord and feeds 16 kHz PCM16 directly into the same
//! RecognitionInput streaming path used by RecognitionService. There is no
//! cross-app IPC or microphone attribution in this mode: keyboard UI, capture,
//! model and inference all live in one Android package/process.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use jni::objects::{GlobalRef, JObject, JShortArray};
use jni::sys::jint;
use jni::JNIEnv;
use once_cell::sync::Lazy;

use crate::engine;

const ERROR_SERVER: i32 = 4;
const ERROR_NO_MATCH: i32 = 7;

struct DirectSession {
    id: u64,
    tx: crossbeam_channel::Sender<engine::RecognitionInput>,
    cancelled: Arc<AtomicBool>,
    finishing: Arc<AtomicBool>,
}

static SESSION: Lazy<Mutex<Option<DirectSession>>> = Lazy::new(|| Mutex::new(None));
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn call_error(env: &mut JNIEnv, target: &JObject, code: i32) {
    let _ = env.call_method(target, "onError", "(I)V", &[code.into()]);
}

fn call_results(env: &mut JNIEnv, target: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(
            target,
            "onResults",
            "(Ljava/lang/String;)V",
            &[(&jtxt).into()],
        );
    }
}

fn clear_session(id: u64) {
    let mut guard = SESSION.lock().unwrap();
    if guard.as_ref().is_some_and(|s| s.id == id) {
        *guard = None;
    }
}

fn cancel_current() {
    let old = SESSION.lock().unwrap().take();
    if let Some(session) = old {
        session.cancelled.store(true, Ordering::SeqCst);
        let _ = session.tx.send(engine::RecognitionInput::Cancel);
    }
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_initNative(
    env: JNIEnv,
    _this: JObject,
    target: JObject,
) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    let Ok(jvm) = env.get_java_vm() else { return; };
    let Ok(target_ref) = env.new_global_ref(&target) else { return; };
    let jvm = Arc::new(jvm);

    std::thread::spawn(move || {
        let _ = engine::ensure_loaded_from_thread(&jvm, &target_ref);
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_startNative(
    env: JNIEnv,
    _this: JObject,
    target: JObject,
) {
    cancel_current();

    let Ok(jvm) = env.get_java_vm() else { return; };
    let Ok(target_ref) = env.new_global_ref(&target) else { return; };
    let jvm = Arc::new(jvm);
    let (tx, rx) = crossbeam_channel::unbounded();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let cancelled = Arc::new(AtomicBool::new(false));
    let finishing = Arc::new(AtomicBool::new(false));

    *SESSION.lock().unwrap() = Some(DirectSession {
        id,
        tx,
        cancelled: cancelled.clone(),
        finishing,
    });

    std::thread::spawn(move || {
        if engine::get_engine().is_none()
            && engine::ensure_loaded_from_thread(&jvm, &target_ref).is_err()
        {
            if !cancelled.load(Ordering::SeqCst) {
                if let Ok(mut env) = jvm.attach_current_thread() {
                    call_error(&mut env, target_ref.as_obj(), ERROR_SERVER);
                }
            }
            clear_session(id);
            return;
        }

        let result = match engine::get_engine() {
            Some(eng) => engine::transcribe_recognition_shared(&eng, &rx),
            None => Err("model unavailable after load".to_string()),
        };

        if cancelled.load(Ordering::SeqCst) {
            clear_session(id);
            return;
        }

        if let Ok(mut env) = jvm.attach_current_thread() {
            match result {
                Ok(Some(text)) if !text.trim().is_empty() => {
                    call_results(&mut env, target_ref.as_obj(), &text)
                }
                Ok(Some(_)) => call_error(&mut env, target_ref.as_obj(), ERROR_NO_MATCH),
                Ok(None) => {}
                Err(e) => {
                    log::error!("HeliBoard direct transcription failed: {}", e);
                    call_error(&mut env, target_ref.as_obj(), ERROR_SERVER);
                }
            }
        }

        clear_session(id);
    });
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_feedAudioNative(
    mut env: JNIEnv,
    _this: JObject,
    samples: JShortArray,
    length: jint,
) {
    if length <= 0 {
        return;
    }

    let tx = {
        let guard = SESSION.lock().unwrap();
        let Some(session) = guard.as_ref() else { return; };
        if session.cancelled.load(Ordering::SeqCst)
            || session.finishing.load(Ordering::SeqCst)
        {
            return;
        }
        session.tx.clone()
    };

    let mut pcm = vec![0i16; length as usize];
    if env.get_short_array_region(&samples, 0, &mut pcm).is_err() {
        return;
    }
    let audio = pcm
        .into_iter()
        .map(|sample| sample as f32 / 32768.0)
        .collect();
    let _ = tx.send(engine::RecognitionInput::Audio(audio));
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_finishNative(
    _env: JNIEnv,
    _this: JObject,
) {
    let tx = {
        let guard = SESSION.lock().unwrap();
        let Some(session) = guard.as_ref() else { return; };
        if session
            .finishing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        session.tx.clone()
    };
    let _ = tx.send(engine::RecognitionInput::Finish);
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_cancelNative(
    _env: JNIEnv,
    _this: JObject,
) {
    cancel_current();
}

#[no_mangle]
pub unsafe extern "system" fn Java_helium314_keyboard_latin_InlineVoiceRecognition_destroyNative(
    _env: JNIEnv,
    _this: JObject,
) {
    cancel_current();
}
