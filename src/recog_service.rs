//! Native backend for `VoiceRecognitionService`, the `android.speech.RecognitionService`
//! implementation that lets *other* keyboards/apps (SwiftKey, Gboard, …) use this app
//! as their offline speech-to-text provider via the system `SpeechRecognizer` API.
//!
//! Unlike the IME / `RecognizeActivity` surfaces (which have their own UI and a manual
//! "tap to stop" control via `voice_session`), a `RecognitionService` has no UI of its
//! own: the calling keyboard expects *us* to decide when the user has finished speaking.
//! So this module adds trailing-silence endpointing on top of the same `engine` model,
//! and finalises automatically (it also honours an explicit `stopListening`/`cancel`).
//!
//! Microphone capture and model compute deliberately live on different threads. The
//! audio callback only measures level and queues PCM; a worker owns the transcribe.cpp
//! session for the request. Streaming-capable models therefore process audio while the
//! user speaks instead of starting a full decode only after endpointing.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jni::objects::{GlobalRef, JClass, JObject};
use jni::JNIEnv;
use once_cell::sync::Lazy;

use crate::engine;
use crate::voice_session::SendStream;

// --- Endpointing / VAD tuning -------------------------------------------------
// These are deliberately simple heuristics on the smoothed mic level. Mic gain
// varies a lot between devices, so they may need tuning; finalisation always
// transcribes whatever was captured, so a mis-tuned threshold only affects the
// auto-stop *timing*, never whether text is returned.
//
/// Absolute smoothed level (0..1) that must be exceeded to count as speech.
const MIN_SPEECH_LEVEL: f32 = 0.12;
/// How far above the running noise floor a level must be to count as speech.
const SPEECH_MARGIN: f32 = 0.08;
/// Trailing silence after speech that triggers auto-finalisation.
const SILENCE_MS: u64 = 1500;
/// If no speech is ever detected, finalise after this long anyway.
const NO_SPEECH_TIMEOUT_MS: u64 = 7000;
/// Hard cap on a single utterance.
const MAX_SESSION_MS: u64 = 60000;
/// Throttle interval for `rmsChanged` UI callbacks.
const LEVEL_UPDATE_MS: u64 = 50;

// Mirror of android.speech.SpeechRecognizer error codes we report.
const ERROR_AUDIO: i32 = 3;
const ERROR_SERVER: i32 = 4;
const ERROR_NO_MATCH: i32 = 7;

/// State shared between the audio callback, endpoint monitor and inference
/// worker. Deliberately does NOT hold the cpal stream, to avoid an Arc cycle
/// (the stream's callback holds an `Arc<Endpoint>`).
struct Endpoint {
    input_tx: crossbeam_channel::Sender<engine::RecognitionInput>,
    sample_count: AtomicUsize,
    last_voice: Mutex<Instant>,
    noise_floor: Mutex<f32>,
    last_level_sent: Mutex<Instant>,
    speech_started: AtomicBool,
    finalized: AtomicBool,
    cancelled: AtomicBool,
    started_at: Instant,
    jvm: Arc<jni::JavaVM>,
    target: GlobalRef,
}

struct Session {
    shared: Arc<Endpoint>,
    stream: Arc<Mutex<Option<SendStream>>>,
}

static SESSION: Lazy<Mutex<Option<Session>>> = Lazy::new(|| Mutex::new(None));

// --- JNI callbacks into VoiceRecognitionService -------------------------------

fn call_void(env: &mut JNIEnv, obj: &JObject, method: &str) {
    let _ = env.call_method(obj, method, "()V", &[]);
}

fn call_rms(env: &mut JNIEnv, obj: &JObject, rms_db: f32) {
    let _ = env.call_method(obj, "onRmsChanged", "(F)V", &[rms_db.into()]);
}

fn call_error(env: &mut JNIEnv, obj: &JObject, code: i32) {
    let _ = env.call_method(obj, "onError", "(I)V", &[code.into()]);
}

fn call_results(env: &mut JNIEnv, obj: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(
            obj,
            "onResults",
            "(Ljava/lang/String;)V",
            &[(&jtxt).into()],
        );
    }
}

// --- JNI entry points ---------------------------------------------------------

/// Called from `onCreate`. Warms up the model in the background so the first
/// recognition after a cold bind is as fast as possible.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_notune_transcribe_VoiceRecognitionService_initNative(
    env: JNIEnv,
    _class: JClass,
    service: JObject,
) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    let jvm = match env.get_java_vm() {
        Ok(vm) => Arc::new(vm),
        Err(_) => return,
    };
    let target_ref = match env.new_global_ref(&service) {
        Ok(r) => r,
        Err(_) => return,
    };

    std::thread::spawn(move || {
        let _ = engine::ensure_loaded_from_thread(&jvm, &target_ref);
    });
}

/// Called from `onStartListening`. Begins microphone capture, starts the model
/// worker, and arms the silence-based endpoint monitor.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_notune_transcribe_VoiceRecognitionService_startListening(
    env: JNIEnv,
    _class: JClass,
    service: JObject,
) {
    let jvm = match env.get_java_vm() {
        Ok(vm) => Arc::new(vm),
        Err(_) => return,
    };
    let target = match env.new_global_ref(&service) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Tear down any session that is still around (e.g. the keyboard called
    // startListening twice without cancel) so its worker can never deliver
    // stale results to this new session.
    {
        let mut guard = SESSION.lock().unwrap();
        if let Some(old) = guard.take() {
            old.shared.cancelled.store(true, Ordering::SeqCst);
            old.shared.finalized.store(true, Ordering::SeqCst);
            let _ = old.shared.input_tx.send(engine::RecognitionInput::Cancel);
            *old.stream.lock().unwrap() = None;
        }
    }

    let (input_tx, input_rx) = crossbeam_channel::unbounded();
    let now = Instant::now();
    let shared = Arc::new(Endpoint {
        input_tx,
        sample_count: AtomicUsize::new(0),
        last_voice: Mutex::new(now),
        noise_floor: Mutex::new(0.0),
        last_level_sent: Mutex::new(now),
        speech_started: AtomicBool::new(false),
        finalized: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
        started_at: now,
        jvm: jvm.clone(),
        target,
    });
    let stream_holder: Arc<Mutex<Option<SendStream>>> = Arc::new(Mutex::new(None));

    // Tell the keyboard we're ready to receive speech.
    {
        let mut env2 = match jvm.attach_current_thread() {
            Ok(e) => e,
            Err(_) => return,
        };
        call_void(&mut env2, shared.target.as_obj(), "onReadyForSpeech");
    }

    // Open the microphone (16 kHz mono, matching the model + voice_session).
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            let mut env2 = jvm.attach_current_thread().unwrap();
            call_error(&mut env2, shared.target.as_obj(), ERROR_AUDIO);
            return;
        }
    };
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    let cb_shared = shared.clone();
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| audio_callback(&cb_shared, data),
        |e| log::error!("RecognitionService stream error: {}", e),
        None,
    );

    match stream {
        Ok(s) => {
            s.play().ok();
            *stream_holder.lock().unwrap() = Some(SendStream(s));
        }
        Err(e) => {
            log::error!("Failed to open microphone: {}", e);
            let mut env2 = jvm.attach_current_thread().unwrap();
            call_error(&mut env2, shared.target.as_obj(), ERROR_AUDIO);
            return;
        }
    }

    // Install the session before either background thread can complete, so
    // clear_session() can reliably identify this request.
    *SESSION.lock().unwrap() = Some(Session {
        shared: shared.clone(),
        stream: stream_holder.clone(),
    });

    // Dedicated inference worker. If model loading is still finishing from the
    // onCreate warm-up, microphone PCM simply queues here; once ready, Unified
    // consumes it incrementally and catches up while speech continues.
    let worker_shared = shared.clone();
    let worker_stream = stream_holder.clone();
    std::thread::spawn(move || inference_worker(worker_shared, worker_stream, input_rx));

    // Endpoint monitor.
    let mon_shared = shared.clone();
    let mon_stream = stream_holder.clone();
    std::thread::spawn(move || endpoint_monitor(mon_shared, mon_stream));
}

/// Called from `onStopListening`: the keyboard asked us to finish now.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_notune_transcribe_VoiceRecognitionService_stopListening(
    _env: JNIEnv,
    _class: JClass,
) {
    let session = SESSION
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| (s.shared.clone(), s.stream.clone()));
    if let Some((shared, stream)) = session {
        std::thread::spawn(move || finalize(shared, stream));
    }
}

/// Called from `onCancel`: discard everything, return nothing.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_notune_transcribe_VoiceRecognitionService_cancelNative(
    _env: JNIEnv,
    _class: JClass,
) {
    let mut guard = SESSION.lock().unwrap();
    if let Some(session) = guard.as_ref() {
        session.shared.cancelled.store(true, Ordering::SeqCst);
        session.shared.finalized.store(true, Ordering::SeqCst);
        let _ = session
            .shared
            .input_tx
            .send(engine::RecognitionInput::Cancel);
        *session.stream.lock().unwrap() = None;
    }
    *guard = None;
}

/// Called from `onDestroy`.
#[no_mangle]
pub unsafe extern "system" fn Java_dev_notune_transcribe_VoiceRecognitionService_destroyNative(
    env: JNIEnv,
    class: JClass,
) {
    Java_dev_notune_transcribe_VoiceRecognitionService_cancelNative(env, class);
}

// --- Audio + endpointing ------------------------------------------------------

fn audio_callback(shared: &Arc<Endpoint>, data: &[f32]) {
    if shared.finalized.load(Ordering::SeqCst) {
        return;
    }

    // Unbounded channel send is non-blocking with respect to model compute. At
    // the current 60 s utterance cap, even a completely stalled worker queues
    // only a few MiB of f32 PCM.
    shared.sample_count.fetch_add(data.len(), Ordering::Relaxed);
    if shared
        .input_tx
        .send(engine::RecognitionInput::Audio(data.to_vec()))
        .is_err()
    {
        return;
    }

    // RMS -> smoothed level in 0..1 (same scaling as voice_session).
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    let rms = (sum / (data.len().max(1) as f32)).sqrt();
    let level = (rms * 6.0).clamp(0.0, 1.0);

    let floor = *shared.noise_floor.lock().unwrap();
    let is_speech = level > MIN_SPEECH_LEVEL && level > floor + SPEECH_MARGIN;

    if is_speech {
        *shared.last_voice.lock().unwrap() = Instant::now();
        // First detected speech -> notify beginningOfSpeech exactly once.
        if shared
            .speech_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            if let Ok(mut env) = shared.jvm.attach_current_thread() {
                call_void(&mut env, shared.target.as_obj(), "onBeginningOfSpeech");
            }
        }
    } else {
        // Slowly adapt the noise floor while no speech is present.
        let mut nf = shared.noise_floor.lock().unwrap();
        *nf = *nf * 0.95 + level * 0.05;
    }

    // Throttled mic-level updates for the keyboard's waveform UI.
    let mut last = shared.last_level_sent.lock().unwrap();
    if last.elapsed() >= Duration::from_millis(LEVEL_UPDATE_MS) {
        *last = Instant::now();
        drop(last);
        if let Ok(mut env) = shared.jvm.attach_current_thread() {
            call_rms(&mut env, shared.target.as_obj(), level * 10.0);
        }
    }
}

fn endpoint_monitor(shared: Arc<Endpoint>, stream: Arc<Mutex<Option<SendStream>>>) {
    loop {
        std::thread::sleep(Duration::from_millis(100));

        if shared.cancelled.load(Ordering::SeqCst) || shared.finalized.load(Ordering::SeqCst) {
            return;
        }

        let elapsed = shared.started_at.elapsed();
        let speech = shared.speech_started.load(Ordering::SeqCst);
        let silence = shared.last_voice.lock().unwrap().elapsed();

        let done = (speech && silence >= Duration::from_millis(SILENCE_MS))
            || elapsed >= Duration::from_millis(MAX_SESSION_MS)
            || (!speech && elapsed >= Duration::from_millis(NO_SPEECH_TIMEOUT_MS));

        if done {
            finalize(shared, stream);
            return;
        }
    }
}

/// Stop capture and tell the already-running model worker that no more PCM is
/// coming. The worker, not this endpoint thread, performs the final stream flush
/// and delivers the transcript.
fn finalize(shared: Arc<Endpoint>, stream: Arc<Mutex<Option<SendStream>>>) {
    if shared
        .finalized
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return; // already finalised/cancelled
    }

    // Stop the microphone first so Finish is guaranteed to be after the final
    // audio chunk in the channel.
    *stream.lock().unwrap() = None;

    let speech = shared.speech_started.load(Ordering::SeqCst);
    let samples = shared.sample_count.load(Ordering::Relaxed);

    let mut env = match shared.jvm.attach_current_thread() {
        Ok(e) => e,
        Err(_) => return,
    };
    let target = shared.target.as_obj();

    if speech {
        call_void(&mut env, target, "onEndOfSpeech");
    }

    // ~0.2s minimum and at least one detected speech interval to bother
    // transcribing. Cancel the model worker so it does not keep waiting.
    if !speech || samples < 3200 {
        shared.cancelled.store(true, Ordering::SeqCst);
        let _ = shared.input_tx.send(engine::RecognitionInput::Cancel);
        call_error(&mut env, target, ERROR_NO_MATCH);
        clear_session(&shared);
        return;
    }

    if shared
        .input_tx
        .send(engine::RecognitionInput::Finish)
        .is_err()
    {
        log::error!("RecognitionService inference worker channel closed before Finish");
        call_error(&mut env, target, ERROR_SERVER);
        clear_session(&shared);
    }
}

// --- Inference worker ---------------------------------------------------------

fn inference_worker(
    shared: Arc<Endpoint>,
    stream: Arc<Mutex<Option<SendStream>>>,
    receiver: crossbeam_channel::Receiver<engine::RecognitionInput>,
) {
    // The service warm-up normally means this is already loaded. Waiting here
    // is still safe: microphone PCM accumulates in the channel while the model
    // finishes loading, rather than delaying capture or blocking its callback.
    if engine::get_engine().is_none()
        && engine::ensure_loaded_from_thread(&shared.jvm, &shared.target).is_err()
    {
        fail_worker(&shared, &stream, ERROR_SERVER);
        return;
    }

    let result = match engine::get_engine() {
        Some(eng_arc) => engine::transcribe_recognition_shared(&eng_arc, &receiver),
        None => Err("model unavailable after load".to_string()),
    };

    // A cancellation may race the last native finalize. Suppress stale results
    // even if the model happened to finish before it consumed the queued Cancel.
    if shared.cancelled.load(Ordering::SeqCst) {
        clear_session(&shared);
        return;
    }

    let mut env = match shared.jvm.attach_current_thread() {
        Ok(e) => e,
        Err(_) => {
            clear_session(&shared);
            return;
        }
    };
    let target = shared.target.as_obj();

    match result {
        Ok(Some(text)) if !text.trim().is_empty() => call_results(&mut env, target, &text),
        Ok(Some(_)) => call_error(&mut env, target, ERROR_NO_MATCH),
        Ok(None) => {}
        Err(e) => {
            log::error!("RecognitionService transcription failed: {}", e);
            call_error(&mut env, target, ERROR_SERVER);
        }
    }

    clear_session(&shared);
}

fn fail_worker(
    shared: &Arc<Endpoint>,
    stream: &Arc<Mutex<Option<SendStream>>>,
    error_code: i32,
) {
    if shared.cancelled.load(Ordering::SeqCst) {
        clear_session(shared);
        return;
    }
    shared.cancelled.store(true, Ordering::SeqCst);
    shared.finalized.store(true, Ordering::SeqCst);
    *stream.lock().unwrap() = None;
    if let Ok(mut env) = shared.jvm.attach_current_thread() {
        call_error(&mut env, shared.target.as_obj(), error_code);
    }
    clear_session(shared);
}

/// Clear the global session, but only if it is still *this* session — a newer
/// `startListening` may already have installed a fresh one.
fn clear_session(shared: &Arc<Endpoint>) {
    let mut guard = SESSION.lock().unwrap();
    if let Some(s) = guard.as_ref() {
        if Arc::ptr_eq(&s.shared, shared) {
            *guard = None;
        }
    }
}
