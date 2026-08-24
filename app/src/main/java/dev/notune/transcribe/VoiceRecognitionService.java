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
import android.os.ParcelFileDescriptor;
import android.os.RemoteException;
import android.speech.RecognitionService;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

/**
 * Exposes the offline transcriber as a system speech-to-text provider via
 * {@link android.speech.RecognitionService}.
 *
 * <p>Preferred Android 13+ path: callers can provide an already-open PCM stream in
 * {@link RecognizerIntent#EXTRA_AUDIO_SOURCE}. In that mode the caller owns microphone
 * permission/AppOps and this background service only consumes PCM and runs inference.
 * This is ideal for an active IME such as HeliBoard.
 *
 * <p>Fallback path: for callers that do not provide audio, the service creates the
 * documented caller-attribution context and opens an attributed AudioRecord itself.
 * Both paths feed the same Rust endpoint/VAD/Parakeet streaming worker.
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
    private InputStream mInjectedInput;
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
            // Keep the documented attribution chain for the fallback microphone path
            // and for RecognitionService's permission bookkeeping on modern Android.
            Context recordingContext = this;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                recordingContext = createContext(
                        new ContextParams.Builder()
                                .setNextAttributionSource(callback.getCallingAttributionSource())
                                .build());
            }

            // Create the native streaming/VAD session before audio starts arriving.
            startListening(this);

            ParcelFileDescriptor injectedSource = getInjectedAudioSource(recognizerIntent);
            if (injectedSource != null) {
                startInjectedAudio(recognizerIntent, injectedSource);
            } else {
                startAudioCapture(recordingContext);
            }
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
        // Stop the producer before sending Finish so no later PCM can race finalization.
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

    private ParcelFileDescriptor getInjectedAudioSource(Intent intent) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return null;
        }
        return intent.getParcelableExtra(
                RecognizerIntent.EXTRA_AUDIO_SOURCE,
                ParcelFileDescriptor.class);
    }

    /** Consume PCM supplied by the SpeechRecognizer caller instead of opening our mic. */
    private void startInjectedAudio(Intent intent, ParcelFileDescriptor source) {
        stopAudioCapture();

        int sampleRate = intent.getIntExtra(
                RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE,
                SAMPLE_RATE);
        int channelCount = intent.getIntExtra(
                RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT,
                1);
        int encoding = intent.getIntExtra(
                RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING,
                AudioFormat.ENCODING_PCM_16BIT);

        if (sampleRate != SAMPLE_RATE
                || channelCount != 1
                || encoding != AudioFormat.ENCODING_PCM_16BIT) {
            try {
                source.close();
            } catch (IOException ignored) {
            }
            throw new IllegalArgumentException(
                    "Unsupported injected audio: " + sampleRate + " Hz, channels="
                            + channelCount + ", encoding=" + encoding);
        }

        ParcelFileDescriptor.AutoCloseInputStream input =
                new ParcelFileDescriptor.AutoCloseInputStream(source);
        mInjectedInput = input;
        mRecording = true;

        onReadyForSpeech();
        Log.i(TAG, "Using caller-provided PCM audio source for recognition");

        mAudioThread = new Thread(() -> readInjectedAudio(input),
                "offline-voice-injected-audio");
        mAudioThread.start();
    }

    /** Read little-endian PCM16 from EXTRA_AUDIO_SOURCE and feed the native stream. */
    private void readInjectedAudio(InputStream input) {
        byte[] bytes = new byte[2048];
        short[] samples = new short[1024];
        int carry = -1;
        boolean reachedEof = false;

        try {
            while (mRecording) {
                int count = input.read(bytes);
                if (count < 0) {
                    reachedEof = true;
                    break;
                }
                if (count == 0) {
                    continue;
                }

                int src = 0;
                int dst = 0;

                if (carry >= 0) {
                    int hi = bytes[src++] & 0xff;
                    samples[dst++] = (short) (carry | (hi << 8));
                    carry = -1;
                }

                while (src + 1 < count) {
                    int lo = bytes[src++] & 0xff;
                    int hi = bytes[src++] & 0xff;
                    samples[dst++] = (short) (lo | (hi << 8));
                }

                if (src < count) {
                    carry = bytes[src] & 0xff;
                }

                if (dst > 0) {
                    feedAudio(samples, dst);
                }
            }
        } catch (IOException e) {
            if (mRecording) {
                Log.w(TAG, "Injected audio source ended with I/O error", e);
            }
        } catch (Throwable t) {
            if (mRecording) {
                Log.e(TAG, "Injected audio feed failed", t);
                mainHandler.post(() -> onError(SpeechRecognizer.ERROR_CLIENT));
            }
            return;
        } finally {
            try {
                input.close();
            } catch (IOException ignored) {
            }
        }

        // EXTRA_AUDIO_SOURCE is defined to end when the caller closes its audio.
        // If Rust did not already endpoint, treat clean EOF as an explicit stop.
        if (reachedEof && mRecording) {
            mRecording = false;
            mainHandler.post(() -> {
                Log.i(TAG, "Caller audio source closed; finalizing recognition");
                try {
                    stopListening();
                } catch (Throwable t) {
                    Log.e(TAG, "finalize after injected EOF failed", t);
                    safeError(SpeechRecognizer.ERROR_CLIENT);
                }
            });
        }
    }

    /** Fallback for SpeechRecognizer callers that do not supply EXTRA_AUDIO_SOURCE. */
    private void startAudioCapture(Context recordingContext) {
        stopAudioCapture();

        int minBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
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

        InputStream injected = mInjectedInput;
        mInjectedInput = null;
        if (injected != null) {
            try {
                injected.close();
            } catch (IOException ignored) {
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
        // Auto-endpointing originates in Rust. Close whichever audio source is active.
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
