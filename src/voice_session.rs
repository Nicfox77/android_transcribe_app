use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jni::objects::{GlobalRef, JObject};
use jni::JNIEnv;

use crate::engine;

// --- Optional auto-stop endpointing (same level heuristics as recog_service) --
/// Absolute smoothed level (0..1) that must be exceeded to count as speech.
const MIN_SPEECH_LEVEL: f32 = 0.12;
/// How far above the running noise floor a level must be to count as speech.
const SPEECH_MARGIN: f32 = 0.08;
/// Trailing silence after speech that triggers auto-stop.
const AUTO_STOP_SILENCE_MS: u64 = 2000;
/// If no speech is ever detected, auto-stop after this long.
const AUTO_STOP_NO_SPEECH_MS: u64 = 8000;

// Parakeet Unified's native buffered-stream mode repeatedly re-encodes a large
// left-context window for every small streaming step. That is useful for true
// incremental hypotheses, but on phone CPUs it can be far slower than the
// model's excellent one-shot benchmark speed. For keyboard dictation we instead
// pre-decode speech continuously in short offline chunks while recording. At
// Stop only the small unprocessed tail remains.
const ROLLING_TARGET_SAMPLES: usize = 5 * 16_000;
const ROLLING_SPLIT_SEARCH_START_SAMPLES: usize = 3 * 16_000;

pub struct SendStream(#[allow(dead_code)] pub cpal::Stream);
unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

/// Speech/silence tracking shared between the audio callback and the
/// auto-stop monitor thread.
struct Endpointing {
    last_voice: Mutex<Instant>,
    noise_floor: Mutex<f32>,
    speech_started: AtomicBool,
}

pub struct VoiceSessionState {
    pub stream: Option<SendStream>,
    pub jvm: Arc<jni::JavaVM>,
    pub target_ref: GlobalRef,
    pub last_level_sent: Arc<Mutex<std::time::Instant>>,
    /// True while the current recording runs; flipped off on stop/cancel so
    /// the auto-stop monitor (if any) exits and the audio callback stops
    /// forwarding PCM.
    pub session_active: Arc<AtomicBool>,
    /// Live microphone -> inference-worker channel for the current recording.
    /// The keyboard worker consumes this continuously and pre-decodes short
    /// chunks before Stop instead of retaining the whole recording.
    pub input_tx: Option<crossbeam_channel::Sender<engine::RecognitionInput>>,
    /// Number of samples forwarded in the current recording. Used only to
    /// reject an empty/failed microphone session before finalization.
    pub sample_count: Arc<AtomicUsize>,
    /// Monotonically increasing session id. Workers check this before emitting
    /// JNI callbacks so a cancelled/replaced recording cannot publish stale
    /// text into a newer one.
    pub session_generation: Arc<AtomicU64>,
}

impl Drop for VoiceSessionState {
    fn drop(&mut self) {
        self.session_active.store(false, Ordering::SeqCst);
        self.stream = None;
        self.session_generation.fetch_add(1, Ordering::SeqCst);
        if let Some(tx) = self.input_tx.take() {
            let _ = tx.send(engine::RecognitionInput::Cancel);
        }
    }
}

fn notify_status(env: &mut JNIEnv, obj: &JObject, msg: &str) {
    if let Ok(jmsg) = env.new_string(msg) {
        let _ = env.call_method(
            obj,
            "onStatusUpdate",
            "(Ljava/lang/String;)V",
            &[(&jmsg).into()],
        );
    }
}

fn notify_level(env: &mut JNIEnv, obj: &JObject, level: f32) {
    let _ = env.call_method(obj, "onAudioLevel", "(F)V", &[level.into()]);
}

fn notify_text(env: &mut JNIEnv, obj: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(
            obj,
            "onTextTranscribed",
            "(Ljava/lang/String;)V",
            &[(&jtxt).into()],
        );
    }
}

fn append_transcript_piece(full: &mut String, piece: &str) {
    let piece = piece.trim();
    if piece.is_empty() {
        return;
    }
    if !full.is_empty() {
        full.push(' ');
    }
    full.push_str(piece);
}

/// Keep the fast offline engine busy *during* microphone capture. Once about
/// five seconds of PCM are buffered, choose the quietest split between 3-5 s,
/// transcribe that prefix, and continue collecting the tail. Because offline
/// inference is several times faster than realtime on the target phone, this
/// worker should stay ahead of the microphone and leave <5 s to process after
/// Stop. Quiet-point splitting avoids most word-boundary artifacts without the
/// repeated 5.6 s left-context cost of Unified's native buffered stream.
fn transcribe_rolling_predecode(
    eng_arc: &Arc<Mutex<engine::Engine>>,
    input_rx: &crossbeam_channel::Receiver<engine::RecognitionInput>,
) -> Result<Option<String>, String> {
    let started = Instant::now();
    let mut pending = Vec::<f32>::with_capacity(ROLLING_TARGET_SAMPLES + 16_000);
    let mut text = String::new();
    let mut total_samples = 0usize;
    let mut decoded_samples = 0usize;
    let mut decode_compute = Duration::ZERO;
    let mut segments = 0usize;

    loop {
        match input_rx.recv() {
            Ok(engine::RecognitionInput::Audio(chunk)) => {
                total_samples += chunk.len();
                pending.extend_from_slice(&chunk);

                while pending.len() >= ROLLING_TARGET_SAMPLES {
                    let split = crate::audio::find_quietest_split(
                        &pending,
                        ROLLING_SPLIT_SEARCH_START_SAMPLES,
                        ROLLING_TARGET_SAMPLES,
                    )
                    .clamp(1, pending.len());

                    let tail = pending.split_off(split);
                    let piece_audio = std::mem::replace(&mut pending, tail);
                    let piece_samples = piece_audio.len();
                    let decode_started = Instant::now();
                    let piece_text = engine::transcribe_shared(eng_arc, piece_audio)?;
                    let elapsed = decode_started.elapsed();

                    decode_compute += elapsed;
                    decoded_samples += piece_samples;
                    segments += 1;
                    append_transcript_piece(&mut text, &piece_text);

                    log::info!(
                        "voice rolling predecode: segment {} = {:.2}s audio in {:.2}s; {:.2}s PCM retained",
                        segments,
                        piece_samples as f64 / 16_000.0,
                        elapsed.as_secs_f64(),
                        pending.len() as f64 / 16_000.0,
                    );
                }
            }
            Ok(engine::RecognitionInput::Finish) => {
                let finalize_started = Instant::now();
                if !pending.is_empty() {
                    let piece_samples = pending.len();
                    let piece_audio = std::mem::take(&mut pending);
                    let decode_started = Instant::now();
                    let piece_text = engine::transcribe_shared(eng_arc, piece_audio)?;
                    let elapsed = decode_started.elapsed();
                    decode_compute += elapsed;
                    decoded_samples += piece_samples;
                    segments += 1;
                    append_transcript_piece(&mut text, &piece_text);
                }

                let audio_secs = total_samples as f64 / 16_000.0;
                let compute_secs = decode_compute.as_secs_f64();
                let speed = if compute_secs > 0.0 {
                    decoded_samples as f64 / 16_000.0 / compute_secs
                } else {
                    0.0
                };
                log::info!(
                    "voice rolling predecode complete: {:.2}s audio, {} segments, {:.2}s total decode ({:.2}x realtime), Stop tail {:.2}s, wall {:.2}s",
                    audio_secs,
                    segments,
                    compute_secs,
                    speed,
                    finalize_started.elapsed().as_secs_f64(),
                    started.elapsed().as_secs_f64(),
                );
                return Ok(Some(text));
            }
            Ok(engine::RecognitionInput::Cancel) | Err(_) => return Ok(None),
        }
    }
}

pub fn init_session(env: JNIEnv, target: JObject) -> VoiceSessionState {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    let vm = env.get_java_vm().expect("Failed to get JavaVM");
    let vm_arc = Arc::new(vm);
    let target_ref = env.new_global_ref(&target).expect("Failed to ref target");

    let state = VoiceSessionState {
        stream: None,
        jvm: vm_arc.clone(),
        target_ref: target_ref.clone(),
        last_level_sent: Arc::new(Mutex::new(std::time::Instant::now())),
        session_active: Arc::new(AtomicBool::new(false)),
        input_tx: None,
        sample_count: Arc::new(AtomicUsize::new(0)),
        session_generation: Arc::new(AtomicU64::new(0)),
    };

    // Load engine in background so a normal first recording usually starts
    // with the model already resident.
    let vm_clone = vm_arc.clone();
    let target_ref_clone = target_ref.clone();
    std::thread::spawn(move || {
        let _ = engine::ensure_loaded_from_thread(&vm_clone, &target_ref_clone);
    });

    state
}

/// Begin microphone capture and start the model worker immediately. Microphone
/// PCM is sent to the worker as it arrives. The keyboard path uses rolling
/// one-shot decoding while recording so it can exploit Parakeet Unified's fast
/// offline throughput without the high repeated-context cost of native buffered
/// streaming. `stop_recording()` therefore only leaves a short tail to decode.
///
/// With `auto_stop` set, a monitor thread watches for trailing silence after
/// speech (or a no-speech timeout) and invokes the Java-side `onAutoStop()`
/// callback, which stops the recording the same way a manual tap would.
pub fn start_recording(mut env: JNIEnv, state: &mut VoiceSessionState, auto_stop: bool) {
    // Cancel any previous worker before replacing the session.
    state.session_active.store(false, Ordering::SeqCst);
    state.stream = None;
    if let Some(tx) = state.input_tx.take() {
        let _ = tx.send(engine::RecognitionInput::Cancel);
    }

    let generation = state.session_generation.fetch_add(1, Ordering::SeqCst) + 1;
    state.sample_count.store(0, Ordering::SeqCst);

    let (input_tx, input_rx) = crossbeam_channel::unbounded();
    state.input_tx = Some(input_tx.clone());

    // Start inference before opening the microphone. If model warm-up is still
    // in progress, PCM simply queues in this unbounded channel and the rolling
    // decoder catches up as soon as loading completes.
    let worker_jvm = state.jvm.clone();
    let worker_target = state.target_ref.clone();
    let worker_generation = state.session_generation.clone();
    std::thread::spawn(move || {
        if engine::get_engine().is_none()
            && engine::ensure_loaded_from_thread(&worker_jvm, &worker_target).is_err()
        {
            return;
        }

        let result = match engine::get_engine() {
            Some(eng_arc) => transcribe_rolling_predecode(&eng_arc, &input_rx),
            None => Err("model unavailable after load".to_string()),
        };

        // A newer recording or cancellation owns the UI now.
        if worker_generation.load(Ordering::SeqCst) != generation {
            return;
        }

        let mut env = match worker_jvm.attach_current_thread() {
            Ok(e) => e,
            Err(_) => return,
        };
        let obj = worker_target.as_obj();

        match result {
            Ok(Some(text)) if !text.trim().is_empty() => {
                notify_status(&mut env, obj, "Ready");
                notify_text(&mut env, obj, &text);
            }
            Ok(Some(_)) => notify_status(&mut env, obj, "Ready"),
            Ok(None) => {}
            Err(e) => {
                log::error!("voice-session transcription failed: {}", e);
                notify_status(&mut env, obj, &format!("Error: {}", e));
            }
        }
    });

    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            if let Some(tx) = state.input_tx.take() {
                let _ = tx.send(engine::RecognitionInput::Cancel);
            }
            state.session_generation.fetch_add(1, Ordering::SeqCst);
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                "Error: no microphone available. Check permissions.",
            );
            return;
        }
    };

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // End any previous session's monitor, then arm a fresh flag.
    let session_active = Arc::new(AtomicBool::new(true));
    state.session_active = session_active.clone();

    let endpoint = if auto_stop {
        Some(Arc::new(Endpointing {
            last_voice: Mutex::new(Instant::now()),
            noise_floor: Mutex::new(0.0),
            speech_started: AtomicBool::new(false),
        }))
    } else {
        None
    };

    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();
    let last_sent = state.last_level_sent.clone();
    let endpoint_cb = endpoint.clone();
    let sample_count = state.sample_count.clone();
    let audio_tx = input_tx.clone();
    let capture_active = session_active.clone();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            if !capture_active.load(Ordering::Relaxed) {
                return;
            }

            // Forward immediately. The send is non-blocking with respect to
            // inference, so model compute can never starve microphone capture.
            sample_count.fetch_add(data.len(), Ordering::Relaxed);
            if audio_tx
                .send(engine::RecognitionInput::Audio(data.to_vec()))
                .is_err()
            {
                return;
            }

            // Compute RMS for the UI and optional endpointing.
            let mut sum = 0.0f32;
            for &x in data {
                sum += x * x;
            }
            let rms = (sum / (data.len().max(1) as f32)).sqrt();
            let level = (rms * 6.0).clamp(0.0, 1.0);

            if let Some(ep) = &endpoint_cb {
                let floor = *ep.noise_floor.lock().unwrap();
                let is_speech = level > MIN_SPEECH_LEVEL && level > floor + SPEECH_MARGIN;
                if is_speech {
                    *ep.last_voice.lock().unwrap() = Instant::now();
                    ep.speech_started.store(true, Ordering::SeqCst);
                } else {
                    // Slowly adapt the noise floor while no speech is present.
                    let mut nf = ep.noise_floor.lock().unwrap();
                    *nf = *nf * 0.95 + level * 0.05;
                }
            }

            // Throttle level updates.
            let mut last = last_sent.lock().unwrap();
            if last.elapsed() >= std::time::Duration::from_millis(50) {
                *last = std::time::Instant::now();
                if let Ok(mut env) = jvm.attach_current_thread() {
                    notify_level(&mut env, target_ref.as_obj(), level);
                }
            }
        },
        |e| log::error!("Stream err: {}", e),
        None,
    );

    match stream {
        Ok(s) => {
            s.play().ok();
            state.stream = Some(SendStream(s));
            notify_status(&mut env, state.target_ref.as_obj(), "Listening...");

            if let Some(ep) = endpoint {
                let jvm = state.jvm.clone();
                let target_ref = state.target_ref.clone();
                let started_at = Instant::now();
                std::thread::spawn(move || loop {
                    std::thread::sleep(Duration::from_millis(100));
                    if !session_active.load(Ordering::SeqCst) {
                        return;
                    }
                    let speech = ep.speech_started.load(Ordering::SeqCst);
                    let silence = ep.last_voice.lock().unwrap().elapsed();
                    let done = (speech
                        && silence >= Duration::from_millis(AUTO_STOP_SILENCE_MS))
                        || (!speech
                            && started_at.elapsed()
                                >= Duration::from_millis(AUTO_STOP_NO_SPEECH_MS));
                    if done {
                        // Claim the session so a simultaneous manual stop and
                        // this monitor can't both fire.
                        if session_active.swap(false, Ordering::SeqCst) {
                            if let Ok(mut env) = jvm.attach_current_thread() {
                                let _ = env.call_method(
                                    target_ref.as_obj(),
                                    "onAutoStop",
                                    "()V",
                                    &[],
                                );
                            }
                        }
                        return;
                    }
                });
            }
        }
        Err(e) => {
            state.session_active.store(false, Ordering::SeqCst);
            if let Some(tx) = state.input_tx.take() {
                let _ = tx.send(engine::RecognitionInput::Cancel);
            }
            state.session_generation.fetch_add(1, Ordering::SeqCst);
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                &format!("Error: failed to open microphone: {}", e),
            );
        }
    }
}

pub fn stop_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    // Stop capture first so no PCM can arrive after Finish.
    state.session_active.store(false, Ordering::SeqCst);
    state.stream = None;

    if state.sample_count.load(Ordering::Relaxed) == 0 {
        if let Some(tx) = state.input_tx.take() {
            let _ = tx.send(engine::RecognitionInput::Cancel);
        }
        state.session_generation.fetch_add(1, Ordering::SeqCst);
        notify_status(
            &mut env,
            state.target_ref.as_obj(),
            "Error: no audio recorded. Check microphone permissions.",
        );
        return;
    }

    // Most of the recording has already been transcribed in rolling chunks.
    // Finish asks the worker to decode only the retained (<5 s) tail and join
    // it with the text produced while the microphone was active.
    notify_status(&mut env, state.target_ref.as_obj(), "Finalizing...");
    match state.input_tx.take() {
        Some(tx) => {
            if tx.send(engine::RecognitionInput::Finish).is_err() {
                notify_status(
                    &mut env,
                    state.target_ref.as_obj(),
                    "Error: transcription worker stopped unexpectedly",
                );
            }
        }
        None => notify_status(
            &mut env,
            state.target_ref.as_obj(),
            "Error: no active transcription session",
        ),
    }
}

pub fn cancel_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    state.session_active.store(false, Ordering::SeqCst);
    state.stream = None;
    if let Some(tx) = state.input_tx.take() {
        let _ = tx.send(engine::RecognitionInput::Cancel);
    }
    state.session_generation.fetch_add(1, Ordering::SeqCst);
    state.sample_count.store(0, Ordering::SeqCst);
    notify_status(&mut env, state.target_ref.as_obj(), "Canceled");
}
