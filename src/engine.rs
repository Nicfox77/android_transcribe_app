//! Loads and serves the speech-to-text engine (transcribe.cpp, GGUF models).
//!
//! The engine is a process-wide singleton behind `Arc<Mutex<..>>`: a
//! transcribe.cpp session may only be used by one thread at a time, which the
//! mutex guarantees. Model selection is read from marker files in filesDir
//! (written by `ModelsActivity`): an imported GGUF if one is selected, the
//! bundled model otherwise — with the bundled model as fallback if the
//! imported one fails to load.

use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};

use jni::objects::{GlobalRef, JObject};
use jni::JNIEnv;

use crate::assets;

/// File in filesDir naming the imported GGUF (a file under `models/`) to use
/// instead of the bundled model. Absent or empty = bundled model.
const ACTIVE_MODEL_FILE: &str = "active_model";
/// File in filesDir with an optional language hint — a locale like `en-US`
/// or `auto`. Absent or empty = let the model autodetect.
const MODEL_LANGUAGE_FILE: &str = "model_language";
/// Marker file in filesDir: when present, models that support translation
/// (e.g. Whisper) translate speech to English instead of transcribing it.
/// Ignored by models without translation support.
const MODEL_TRANSLATE_FILE: &str = "model_translate";
/// Optional file in filesDir with the CPU thread count for inference.
/// Absent/invalid/0 = default (all cores).
const MODEL_THREADS_FILE: &str = "model_threads";

/// Longest audio passed to the model in one run (60 s). Offline conformer
/// models use full self-attention, whose cost grows quadratically with input
/// length — an unbounded shared audio file would exhaust memory on a phone.
/// Longer input is split at quiet points and the texts joined.
const MAX_RUN_SAMPLES: usize = 60 * 16_000;
/// When splitting, search this far back from the hard boundary for the
/// quietest point so words aren't cut mid-syllable.
const SPLIT_SEARCH_SAMPLES: usize = 10 * 16_000;

/// Audio/control messages sent from Android microphone capture to the model
/// worker. Keeping inference off the realtime audio callback prevents model
/// compute from starving capture and lets streaming models stay caught up while
/// the user is still speaking.
pub enum RecognitionInput {
    Audio(Vec<f32>),
    Finish,
    Cancel,
}

/// A loaded transcribe.cpp session plus the options applied to every run.
pub struct Engine {
    session: transcribe_cpp::Session,
    language: Option<String>,
    task: transcribe_cpp::Task,
    /// Family-specific decode options attached to every run; `None` for
    /// models that don't take the whisper run extension.
    run_ext: Option<transcribe_cpp::RunExtension>,
    /// True when the loaded model accepts Parakeet's chunked-attention buffered
    /// streaming extension (currently Parakeet Unified EN 0.6B).
    buffered_streaming: bool,
    /// Status reported once loading succeeded; carries a warning when the
    /// translate setting can't do what the user expects with this model.
    ready_status: &'static str,
}

impl Engine {
    fn load(
        model_path: &Path,
        language: Option<String>,
        translate: bool,
        threads: i32,
    ) -> Result<Engine, String> {
        if !model_path.is_file() {
            return Err(format!("model file not found: {}", model_path.display()));
        }
        let model = transcribe_cpp::Model::load(model_path).map_err(|e| e.to_string())?;
        // Translation is gated on the model's capabilities: models without it
        // (e.g. Parakeet) silently keep transcribing, so the setting can stay
        // on while switching models.
        let task = if translate && model.capabilities().supports_translate {
            transcribe_cpp::Task::Translate
        } else {
            if translate {
                log::info!("translate requested but unsupported by this model; transcribing");
            }
            transcribe_cpp::Task::Transcribe
        };
        // Silently doing something other than what the translate switch says
        // looks like a bug (still-untranslated subtitles), so say it in the
        // status. Whisper Turbo is special-cased: it advertises translation
        // but was distilled without translation data — verified on-device to
        // keep transcribing German as German with task=Translate.
        let ready_status = if translate && task == transcribe_cpp::Task::Transcribe {
            "Ready (this model can't translate)"
        } else if translate && model.variant().contains("turbo") {
            "Ready (note: Whisper Turbo translates poorly; use another Whisper)"
        } else {
            "Ready"
        };
        // Whisper's stock recipe re-decodes a chunk at up to five higher
        // temperatures when its quality gates fail, so one noisy chunk can
        // cost several full decodes. For an interactive app a single greedy
        // pass is the better trade: worst case is a worse line of text, not
        // a multiplied wait. temperature_inc = 0 turns the retry ladder off;
        // models that don't take the whisper run extension are unaffected.
        let run_ext = if model.accepts_ext(
            transcribe_cpp::ExtSlot::Run,
            transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_WHISPER_RUN,
        ) {
            Some(transcribe_cpp::RunExtension::Whisper(
                transcribe_cpp::WhisperRunOptions {
                    temperature_inc: Some(0.0),
                    ..Default::default()
                },
            ))
        } else {
            None
        };
        let buffered_streaming = model.accepts_ext(
            transcribe_cpp::ExtSlot::Stream,
            transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
        );

        log::info!(
            "engine: {} threads, task {:?}, single-pass decode: {}, parakeet buffered stream: {}",
            threads,
            task,
            run_ext.is_some(),
            buffered_streaming
        );
        let options = transcribe_cpp::SessionOptions {
            n_threads: threads,
            ..Default::default()
        };
        let session = model.session_with(&options).map_err(|e| e.to_string())?;
        Ok(Engine {
            session,
            language,
            task,
            run_ext,
            buffered_streaming,
            ready_status,
        })
    }

    /// Transcribes 16 kHz mono f32 samples to text. Input longer than
    /// [`MAX_RUN_SAMPLES`] is transcribed in quiet-point chunks.
    pub fn transcribe(&mut self, samples: Vec<f32>) -> Result<String, String> {
        if samples.len() <= MAX_RUN_SAMPLES {
            return self.run(&samples);
        }

        let mut text = String::new();
        let mut rest: &[f32] = &samples;
        while !rest.is_empty() {
            let take = if rest.len() <= MAX_RUN_SAMPLES {
                rest.len()
            } else {
                crate::audio::find_quietest_split(
                    rest,
                    MAX_RUN_SAMPLES - SPLIT_SEARCH_SAMPLES,
                    MAX_RUN_SAMPLES,
                )
            };
            let piece = self.run(&rest[..take])?;
            let piece = piece.trim();
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            rest = &rest[take..];
        }
        Ok(text)
    }

    /// Drives one Android recognition request from microphone chunks. Models
    /// that advertise Parakeet buffered streaming are processed continuously;
    /// every other model keeps the existing record-then-transcribe behavior.
    pub fn transcribe_recognition(
        &mut self,
        receiver: &crossbeam_channel::Receiver<RecognitionInput>,
    ) -> Result<Option<String>, String> {
        if self.buffered_streaming {
            self.transcribe_buffered_stream(receiver)
        } else {
            self.transcribe_channel_batch(receiver, Vec::new())
        }
    }

    /// Existing batch behavior, but fed through the same microphone channel as
    /// the streaming path. `prefix` is used to recover safely if a streaming
    /// feed/finalize fails after some audio has already been received.
    fn transcribe_channel_batch(
        &mut self,
        receiver: &crossbeam_channel::Receiver<RecognitionInput>,
        mut prefix: Vec<f32>,
    ) -> Result<Option<String>, String> {
        loop {
            match receiver.recv() {
                Ok(RecognitionInput::Audio(chunk)) => prefix.extend_from_slice(&chunk),
                Ok(RecognitionInput::Finish) => return self.transcribe(prefix).map(Some),
                Ok(RecognitionInput::Cancel) | Err(_) => return Ok(None),
            }
        }
    }

    /// Parakeet Unified buffered streaming at the 1.12 s operating point:
    /// 5.60 s left context + 560 ms current chunk + 560 ms right context.
    ///
    /// We intentionally use `OnFinalize`: inference stays current while the user
    /// speaks, but Android only receives one stable final transcript. A copy of
    /// received PCM is retained for this short RecognitionService session so a
    /// native streaming error can fall back to the proven one-shot path without
    /// losing dictated audio.
    fn transcribe_buffered_stream(
        &mut self,
        receiver: &crossbeam_channel::Receiver<RecognitionInput>,
    ) -> Result<Option<String>, String> {
        let run_opts = transcribe_cpp::RunOptions {
            // Unified is English-only and does not need a locale hint. Avoid a
            // saved en-US/en-GB hint making stream_begin reject an otherwise
            // valid model.
            language: None,
            task: self.task,
            family: self.run_ext.clone(),
            ..Default::default()
        };
        let stream_opts = transcribe_cpp::StreamOptions {
            commit_policy: transcribe_cpp::CommitPolicy::OnFinalize,
            family: Some(transcribe_cpp::StreamExtension::ParakeetBuffered(
                transcribe_cpp::ParakeetBufferedStreamOptions {
                    left_ms: Some(5600),
                    chunk_ms: Some(560),
                    right_ms: Some(560),
                },
            )),
            ..Default::default()
        };

        // If model capability detection says buffered streaming is supported,
        // stream creation should succeed. Propagate a begin failure instead of
        // borrowing the same Session a second time from the Err arm; feed and
        // finalize failures below still retain the one-shot fallback.
        let mut stream = self
            .session
            .stream(&run_opts, &stream_opts)
            .map_err(|e| format!("buffered stream begin failed: {}", e))?;

        let started = std::time::Instant::now();
        let mut compute = std::time::Duration::ZERO;
        let mut fallback_audio = Vec::new();

        loop {
            match receiver.recv() {
                Ok(RecognitionInput::Audio(chunk)) => {
                    fallback_audio.extend_from_slice(&chunk);
                    let t = std::time::Instant::now();
                    if let Err(e) = stream.feed(&chunk) {
                        compute += t.elapsed();
                        log::error!(
                            "buffered stream feed failed after {:.1}s audio: {}; falling back to batch",
                            fallback_audio.len() as f64 / 16_000.0,
                            e
                        );
                        stream.reset();
                        drop(stream);
                        return self.transcribe_channel_batch(receiver, fallback_audio);
                    }
                    compute += t.elapsed();
                }
                Ok(RecognitionInput::Finish) => {
                    let finalize_started = std::time::Instant::now();
                    match stream.finalize() {
                        Ok(update) => {
                            let finalize_elapsed = finalize_started.elapsed();
                            let snapshot = stream.text();
                            let text = if !snapshot.full.trim().is_empty() {
                                snapshot.full.clone()
                            } else if !snapshot.committed.trim().is_empty()
                                || !snapshot.tentative.trim().is_empty()
                            {
                                format!("{}{}", snapshot.committed, snapshot.tentative)
                            } else {
                                String::new()
                            };
                            let audio_secs = fallback_audio.len() as f64 / 16_000.0;
                            let compute_secs = compute.as_secs_f64();
                            let rtf = if audio_secs > 0.0 {
                                compute_secs / audio_secs
                            } else {
                                0.0
                            };
                            log::info!(
                                "buffered streaming: {:.1}s audio, feed compute {:.2}s (RTF {:.2}), finalize {:.2}s, wall {:.2}s, buffered {}ms, full={} chars, committed={} chars, tentative={} chars",
                                audio_secs,
                                compute_secs,
                                rtf,
                                finalize_elapsed.as_secs_f64(),
                                started.elapsed().as_secs_f64(),
                                update.buffered_ms,
                                snapshot.full.len(),
                                snapshot.committed.len(),
                                snapshot.tentative.len()
                            );
                            if text.trim().is_empty() && !fallback_audio.is_empty() {
                                log::warn!(
                                    "buffered stream finalized empty; retrying retained PCM in batch mode"
                                );
                                drop(stream);
                                return self.transcribe(fallback_audio).map(Some);
                            }
                            return Ok(Some(text));
                        }
                        Err(e) => {
                            log::error!(
                                "buffered stream finalize failed ({}); falling back to batch",
                                e
                            );
                            stream.reset();
                            drop(stream);
                            return self.transcribe(fallback_audio).map(Some);
                        }
                    }
                }
                Ok(RecognitionInput::Cancel) | Err(_) => {
                    stream.reset();
                    return Ok(None);
                }
            }
        }
    }

    /// One model run. A rejected language hint is degraded instead of
    /// failing the transcription: `de-DE` retries as `de`, then as no hint
    /// (each model knows a different set of tags — e.g. Parakeet v3 takes
    /// locales/short codes, English-only models take none). The degraded
    /// value is kept so later runs skip the rejected attempts.
    fn run(&mut self, samples: &[f32]) -> Result<String, String> {
        loop {
            let opts = transcribe_cpp::RunOptions {
                language: self.language.clone(),
                task: self.task,
                family: self.run_ext.clone(),
                ..Default::default()
            };
            match self.session.run(samples, &opts) {
                Ok(t) => return Ok(t.text),
                Err(transcribe_cpp::Error::Unsupported(msg)) if self.language.is_some() => {
                    let lang = self.language.take().unwrap();
                    self.language = lang.split_once('-').map(|(primary, _)| primary.to_string());
                    log::warn!(
                        "language hint '{}' rejected ({}); retrying with {:?}",
                        lang,
                        msg,
                        self.language
                    );
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }
}

/// Holds the loaded engine singleton.
static GLOBAL_ENGINE: Lazy<Mutex<Option<Arc<Mutex<Engine>>>>> =
    Lazy::new(|| Mutex::new(None));

/// Loading coordination state + condvar for waiters.
static LOAD_STATE: Lazy<(Mutex<LoadState>, Condvar)> =
    Lazy::new(|| (Mutex::new(LoadState::Idle), Condvar::new()));

#[derive(Debug, Clone, PartialEq)]
enum LoadState {
    /// No load in progress
    Idle,
    /// A thread is currently loading
    Loading,
    /// Loading completed successfully
    Done,
    /// Loading failed
    Failed(String),
}

pub fn get_engine() -> Option<Arc<Mutex<Engine>>> {
    GLOBAL_ENGINE.lock().unwrap().clone()
}

/// Runs a transcription on the shared engine. Two layers of hardening keep a
/// single bad run from freezing every later one (the symptom would be an IME
/// stuck at "Processing" until its process dies, since notify callbacks stop
/// coming): a panic anywhere in the engine stack is caught and surfaced as a
/// normal error, and a lock poisoned by an earlier panic is recovered instead
/// of propagating the poison forever.
pub fn transcribe_shared(engine: &Arc<Mutex<Engine>>, samples: Vec<f32>) -> Result<String, String> {
    let audio_secs = samples.len() as f64 / 16_000.0;
    let started = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.transcribe(samples)
    }))
    .unwrap_or_else(|_| {
        log::error!("transcription panicked; reporting as error");
        Err("transcription failed unexpectedly, please try again".to_string())
    });
    log::info!(
        "transcribed {:.1}s audio in {:.2}s",
        audio_secs,
        started.elapsed().as_secs_f64()
    );
    result
}

/// Runs one RecognitionService session, holding the engine lock for the life of
/// a native stream. Non-streaming models simply wait for Finish before running
/// the existing one-shot path.
pub fn transcribe_recognition_shared(
    engine: &Arc<Mutex<Engine>>,
    receiver: &crossbeam_channel::Receiver<RecognitionInput>,
) -> Result<Option<String>, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.transcribe_recognition(receiver)
    }))
    .unwrap_or_else(|_| {
        log::error!("recognition streaming worker panicked; reporting as error");
        Err("recognition failed unexpectedly, please try again".to_string())
    })
}

pub fn is_engine_loaded() -> bool {
    GLOBAL_ENGINE.lock().unwrap().is_some()
}

/// Drops the loaded engine and clears the load state so the next
/// `ensure_loaded*` call reloads with the current model selection. Waits for
/// an in-flight load to finish first, so a reload can't race a load of the
/// previous selection. Call from a background thread.
pub fn reset() {
    let (lock, cvar) = &*LOAD_STATE;
    let mut state = lock.lock().unwrap();
    while *state == LoadState::Loading {
        state = cvar.wait(state).unwrap();
    }
    *GLOBAL_ENGINE.lock().unwrap() = None;
    *state = LoadState::Idle;
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

/// Ensures the engine is loaded. Safe to call from multiple threads
/// concurrently; reports status via the target's `onStatusUpdate` callback.
///
/// Convenience wrapper around [`ensure_loaded_from_thread`] for JNI entry
/// points that already hold an attached `JNIEnv`.
pub fn ensure_loaded(env: &mut JNIEnv, context: &JObject) -> Result<(), String> {
    let jvm = Arc::new(env.get_java_vm().map_err(|e| e.to_string())?);
    let context_ref = env.new_global_ref(context).map_err(|e| e.to_string())?;
    ensure_loaded_from_thread(&jvm, &context_ref)
}

/// Ensures the engine is loaded. Safe to call from multiple threads concurrently.
///
/// - If already loaded, returns immediately.
/// - If another thread is loading, waits for it to finish.
/// - If no one is loading, this thread takes ownership of loading.
/// - If a previous load failed, retries.
///
/// Reports status via the target's `onStatusUpdate` JNI callback.
pub fn ensure_loaded_from_thread(
    jvm: &Arc<jni::JavaVM>,
    target_ref: &GlobalRef,
) -> Result<(), String> {
    let notify = |msg: &str| {
        if let Ok(mut env) = jvm.attach_current_thread() {
            notify_status(&mut env, target_ref.as_obj(), msg);
        }
    };

    // Fast path: already loaded
    if is_engine_loaded() {
        notify("Ready");
        return Ok(());
    }

    let (lock, cvar) = &*LOAD_STATE;
    let mut state = lock.lock().unwrap();

    // Re-check under lock
    if is_engine_loaded() {
        notify("Ready");
        return Ok(());
    }

    match &*state {
        LoadState::Loading => {
            // Another thread is loading — wait for it
            notify("Waiting for model...");
            while *state == LoadState::Loading {
                state = cvar.wait(state).unwrap();
            }
            drop(state);

            if is_engine_loaded() {
                notify("Ready");
                Ok(())
            } else {
                let msg = "Model failed to load".to_string();
                notify(&format!("Error: {}", msg));
                Err(msg)
            }
        }
        LoadState::Done => {
            notify("Ready");
            Ok(())
        }
        LoadState::Idle | LoadState::Failed(_) => {
            // We take ownership of loading (retry on previous failure)
            *state = LoadState::Loading;
            drop(state);

            let result = if let Ok(mut env) = jvm.attach_current_thread() {
                do_load(&mut env, target_ref.as_obj())
            } else {
                Err("Failed to attach JNI thread".to_string())
            };

            let mut state = lock.lock().unwrap();
            match &result {
                Ok(()) => *state = LoadState::Done,
                Err(msg) => *state = LoadState::Failed(msg.clone()),
            }
            cvar.notify_all();
            result
        }
    }
}

/// Inference thread count: the number of performance cores, capped at 4.
///
/// Phone SoCs are heterogeneous: fast cores paired with slow efficiency
/// cores. ggml synchronizes all threads after each operation, so a thread
/// on a slow core stalls the whole pool, and using every core is slower
/// than using only the fast ones. Cores are classified by their maximum
/// frequency from sysfs: within 70% of the fastest core counts as fast.
/// The count is capped at 4 because grabbing every fast core leaves none
/// for the rest of the system (audio pipeline, the app playing the sound),
/// and any preempted worker stalls the pool at the next op barrier; past
/// 4 threads the matmuls are memory-bound on phone-class SoCs anyway, and
/// more threads mainly build up heat. If sysfs is unreadable, falls back
/// to a conservative 4. The `model_threads` config file (user-settable in
/// the Models screen) overrides the heuristic.
fn performance_core_count() -> i32 {
    let mut freqs: Vec<u64> = Vec::new();
    for i in 0..64 {
        let path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
        match std::fs::read_to_string(&path) {
            Ok(s) => match s.trim().parse::<u64>() {
                Ok(f) => freqs.push(f),
                Err(_) => break,
            },
            Err(_) => break,
        }
    }
    match freqs.iter().max() {
        Some(&max) if max > 0 => {
            let fast = freqs.iter().filter(|&&f| f * 10 >= max * 7).count();
            let threads = fast.clamp(1, 4);
            log::info!(
                "cpu clusters {:?} kHz -> {} performance cores -> {} threads",
                freqs,
                fast,
                threads
            );
            threads as i32
        }
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(4) as i32,
    }
}

/// CPU features this build's ggml kernels require (see GGML_CPU_ARM_ARCH in
/// the gradle build): dot-product and half-precision SIMD, present on arm64
/// cores since ~2018. Without this check, an older CPU would crash with an
/// illegal instruction mid-inference instead of showing an error.
#[cfg(target_arch = "aarch64")]
fn check_cpu_features() -> Result<(), String> {
    const HWCAP_ASIMHP: libc::c_ulong = 1 << 10; // FEAT_FP16 (asimdhp)
    const HWCAP_ASIMDDP: libc::c_ulong = 1 << 20; // FEAT_DotProd (asimddp)
    let hwcap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    if hwcap & HWCAP_ASIMDDP == 0 || hwcap & HWCAP_ASIMHP == 0 {
        return Err(
            "this device's CPU is too old for this app version (needs arm64 \
             dotprod/fp16, available on phones from ~2018 on)"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn check_cpu_features() -> Result<(), String> {
    Ok(())
}

/// Reads a single-value config file (trimmed); `None` if absent or empty.
fn read_config(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Performs the model load: the selected imported GGUF if any (falling back
/// to the bundled model on failure), otherwise the bundled model.
fn do_load(env: &mut JNIEnv, context: &JObject) -> Result<(), String> {
    if let Err(msg) = check_cpu_features() {
        notify_status(env, context, &format!("Error: {}", msg));
        return Err(msg);
    }

    let files_dir = assets::files_dir(env, context).map_err(|e| {
        let msg = format!("Failed to resolve filesDir: {}", e);
        notify_status(env, context, &format!("Error: {}", msg));
        msg
    })?;
    // Absent/empty or "auto" = no hint (the model's automatic mode). A tag
    // the model doesn't know is degraded per run — see Engine::run.
    let language = read_config(&files_dir.join(MODEL_LANGUAGE_FILE))
        .filter(|l| !l.eq_ignore_ascii_case("auto"));
    let translate = files_dir.join(MODEL_TRANSLATE_FILE).exists();
    let threads = read_config(&files_dir.join(MODEL_THREADS_FILE))
        .and_then(|s| s.parse::<i32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(performance_core_count);

    if let Some(name) = read_config(&files_dir.join(ACTIVE_MODEL_FILE)) {
        let path = files_dir.join("models").join(&name);
        notify_status(env, context, &format!("Loading model {}...", name));
        match Engine::load(&path, language.clone(), translate, threads) {
            Ok(engine) => {
                let status = engine.ready_status;
                *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
                notify_status(env, context, status);
                return Ok(());
            }
            Err(e) => {
                log::error!("Imported model {} failed to load: {}", path.display(), e);
                notify_status(
                    env,
                    context,
                    &format!("Error loading {}: {} — using built-in model", name, e),
                );
                // fall through to the bundled model
            }
        }
    }

    notify_status(env, context, "Checking assets...");

    let path = assets::extract_builtin_model(env, context).map_err(|e| {
        let msg = format!("Asset error: {}", e);
        notify_status(env, context, &format!("Error: {}", msg));
        msg
    })?;

    notify_status(env, context, "Loading model...");

    match Engine::load(&path, language, translate, threads) {
        Ok(engine) => {
            let status = engine.ready_status;
            *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
            notify_status(env, context, status);
            Ok(())
        }
        Err(e) => {
            // Load failed — likely corrupt/incomplete extraction. Invalidate
            // it so the next attempt re-extracts from the APK.
            assets::invalidate_builtin_model(&path);
            let msg = format!("Model error: {}", e);
            notify_status(env, context, &format!("Error: {}", msg));
            Err(msg)
        }
    }
}
