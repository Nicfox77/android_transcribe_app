package dev.notune.transcribe;

import android.content.Context;
import android.content.ContextParams;
import android.content.Intent;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.RemoteException;
import android.speech.RecognitionService;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.util.ArrayList;

/**
 * Exposes the offline transcriber as a system speech-to-text provider via
 * {@link android.speech.RecognitionService}.
 *
 * <p>For microphone recognition Android expects a RecognitionService to create
 * a caller-attribution context inside {@link #onStartListening(Intent, Callback)}
 * and open the microphone through that context. This is important for IME callers:
 * Android checks the keyboard's RECORD_AUDIO identity and records this service as
 * a proxy. The previous native/cpal capture did not create that attribution chain,
 * which caused ERROR_INSUFFICIENT_PERMISSIONS on modern Android/GrapheneOS.
 *
 * <p>AudioRecord captures 16 kHz mono PCM16 here and forwards small chunks to the
 * existing Rust endpoint/VAD/Parakeet streaming worker. Model inference remains native.
 */
public class VoiceRecognitionService extends RecognitionService {

    private static final String TAG = "OfflineVoiceInput";
    private static final int SAMPLE_RATE = 16000;

    static {
        try {
            System.loadLibrary("c++_shared");
            System.loadLibrary("android_transcribe_app");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load native libraries", e);
        }
    }

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private Callback mCallback;
    private volatile boolean mRecording;
    private AudioRecord mAudioRecord;
    private Thread mAudioThread;

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            initNative(this);
        } catch (Throwable t) {
            Log.e(TAG, "initNative failed", t);
        }
    }

    @Override
    protected void onStartListening(Intent recognizerIntent, Callback callback) {
        mCallback = callback;

        try {
            // RecognitionService.createContext() notices this caller attribution
            // synchronously on its handler thread. That is also what Android's
            // permission bookkeeping expects before onStartListening returns.
            Context recordingContext = this;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                recordingContext = createContext(
                        new ContextParams.Builder()
                                .setNextAttributionSource(callback.getCallingAttributionSource())
                                .build());
            }

            // Create the native streaming/VAD session first. Java then owns the
            // attributed microphone and forwards PCM to that worker.
            startListening(this);
            startAudioCapture(recordingContext);
        } catch (Throwable t) {
            Log.e(TAG, "startListening failed", t);
            stopAudioCapture();
            try {
                cancelNative();
            } catch (Throwable ignored) {
            }
            safeError(SpeechRecognizer.ERROR_AUDIO);
        }
    }

    @Override
    protected void onStopListening(Callback callback) {
        // Stop capture before sending Finish so no later PCM can race finalization.
        stopAudioCapture();
        try {
            stopListening();
        } catch (Throwable t) {
            Log.e(TAG, "stopListening failed", t);
        }
    }

    @Override
    protected void onCancel(Callback callback) {
        stopAudioCapture();
        try {
            cancelNative();
        } catch (Throwable t) {
            Log.e(TAG, "cancel failed", t);
        }
    }

    @Override
    public void onDestroy() {
        stopAudioCapture();
        try {
            destroyNative();
        } catch (Throwable t) {
            Log.e(TAG, "destroyNative failed", t);
        }
        super.onDestroy();
    }

    private void startAudioCapture(Context recordingContext) {
        stopAudioCapture();

        int minBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        // Give AudioRecord roughly one second of internal buffering while JNI/model
        // processing runs on separate threads. Reads themselves remain small/low-latency.
        int bufferBytes = Math.max(minBytes > 0 ? minBytes : 0, SAMPLE_RATE * 2);

        AudioRecord.Builder builder = new AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build())
                .setBufferSizeInBytes(bufferBytes);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            builder.setContext(recordingContext);
        }

        AudioRecord recorder = builder.build();
        if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
            recorder.release();
            throw new IllegalStateException("AudioRecord failed to initialize");
        }

        mAudioRecord = recorder;
        mRecording = true;
        recorder.startRecording();
        if (recorder.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
            stopAudioCapture();
            throw new IllegalStateException("AudioRecord failed to start");
        }

        onReadyForSpeech();
        Log.i(TAG, "Attributed AudioRecord started for recognition");

        mAudioThread = new Thread(() -> {
            // 1024 samples ~= 64 ms at 16 kHz: small enough for responsive VAD and
            // partial streaming without putting inference on the audio thread.
            short[] buffer = new short[1024];
            while (mRecording) {
                AudioRecord current = mAudioRecord;
                if (current == null) break;

                int count = current.read(buffer, 0, buffer.length, AudioRecord.READ_BLOCKING);
                if (count > 0) {
                    try {
                        feedAudio(buffer, count);
                    } catch (Throwable t) {
                        Log.e(TAG, "feedAudio failed", t);
                        mainHandler.post(() -> onError(SpeechRecognizer.ERROR_CLIENT));
                        break;
                    }
                } else if (count < 0 && mRecording) {
                    Log.e(TAG, "AudioRecord read failed: " + count);
                    mainHandler.post(() -> onError(SpeechRecognizer.ERROR_AUDIO));
                    break;
                }
            }
        }, "offline-voice-attributed-capture");
        mAudioThread.start();
    }

    private synchronized void stopAudioCapture() {
        mRecording = false;

        AudioRecord recorder = mAudioRecord;
        mAudioRecord = null;
        if (recorder != null) {
            try {
                recorder.stop();
            } catch (Throwable ignored) {
            }
            try {
                recorder.release();
            } catch (Throwable ignored) {
            }
        }

        Thread thread = mAudioThread;
        mAudioThread = null;
        if (thread != null && thread != Thread.currentThread()) {
            try {
                thread.join(250);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    // --- Callbacks invoked from native code (any thread) ---------------------

    public void onReadyForSpeech() {
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            try {
                cb.readyForSpeech(new Bundle());
            } catch (RemoteException ignored) {
            }
        });
    }

    public void onBeginningOfSpeech() {
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            try {
                cb.beginningOfSpeech();
            } catch (RemoteException ignored) {
            }
        });
    }

    public void onRmsChanged(float rmsdB) {
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            try {
                cb.rmsChanged(rmsdB);
            } catch (RemoteException ignored) {
            }
        });
    }

    public void onEndOfSpeech() {
        // Auto-endpointing originates in Rust. Close the recorder immediately so
        // no more samples can arrive while the stream is being finalized.
        stopAudioCapture();
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            try {
                cb.endOfSpeech();
            } catch (RemoteException ignored) {
            }
        });
    }

    /** Deliver a revisable streaming hypothesis without ending recognition. */
    public void onPartialResults(String text) {
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null || text == null || text.trim().isEmpty()) return;
            ArrayList<String> hypotheses = new ArrayList<>();
            hypotheses.add(text);
            Bundle bundle = new Bundle();
            bundle.putStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION, hypotheses);
            try {
                cb.partialResults(bundle);
            } catch (RemoteException ignored) {
            }
        });
    }

    public void onResults(String text) {
        stopAudioCapture();
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            ArrayList<String> hypotheses = new ArrayList<>();
            hypotheses.add(text);
            Bundle bundle = new Bundle();
            bundle.putStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION, hypotheses);
            try {
                cb.results(bundle);
            } catch (RemoteException ignored) {
            }
            mCallback = null;
        });
    }

    public void onError(int errorCode) {
        stopAudioCapture();
        mainHandler.post(() -> {
            Callback cb = mCallback;
            if (cb == null) return;
            try {
                cb.error(errorCode);
            } catch (RemoteException ignored) {
            }
            mCallback = null;
        });
    }

    /** Invoked by the shared engine loader during model warm-up; UI-less here. */
    public void onStatusUpdate(String status) {
        Log.d(TAG, "engine: " + status);
    }

    private void safeError(int errorCode) {
        Callback cb = mCallback;
        if (cb == null) return;
        try {
            cb.error(errorCode);
        } catch (RemoteException ignored) {
        }
        mCallback = null;
    }

    // --- Native methods (implemented in src/recog_service.rs) ----------------

    private native void initNative(VoiceRecognitionService service);
    private native void startListening(VoiceRecognitionService service);
    private native void feedAudio(short[] samples, int length);
    private native void stopListening();
    private native void cancelNative();
    private native void destroyNative();
}
