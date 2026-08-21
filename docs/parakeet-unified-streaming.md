# Parakeet Unified streaming voice input

This fork adds true incremental inference to Android's `RecognitionService` when the active transcribe.cpp model advertises the Parakeet buffered-streaming extension.

## Recommended model

Use `handy-computer/parakeet-unified-en-0.6b-gguf` with:

- File: `parakeet-unified-en-0.6b-Q4_K_M.gguf`
- Size: about 477 MB
- SHA-256: `a8bf3de2b393bd14ead5a858c3748d5e3b07a20fdeabdd3b498fba4f463fa929`
- Download: `https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q4_K_M.gguf`

For the normal app build, download the GGUF on Android, open **Models** in Offline Voice Input, import it, and select it as the active model. The normal build keeps Parakeet TDT v3 bundled, and non-streaming models continue to use the existing batch transcription path.

## Side-by-side test APK

The feature branch also has a `Build Unified streaming test APK` GitHub Actions workflow. Its debug artifact:

- bundles Unified Q4_K_M directly, so no separate model import is needed;
- uses application id `dev.notune.transcribe.unifiedtest`, so it can be installed beside the normal app;
- labels itself **Offline Voice Input (Unified Test)**;
- forces a versioned bundled-model extraction marker so a stale V3 extraction cannot be reused.

This packaging change is applied only inside the test workflow; it does not alter the normal app artifact.

## Streaming profile

The RecognitionService uses the 1.12-second Parakeet Unified profile:

- left context: 5600 ms
- chunk: 560 ms
- right context: 560 ms
- commit policy: `OnFinalize`

Audio is fed to the model continuously while the microphone is active, but Android receives only the final stable transcript. This is meant to remove most of the post-recording wait without showing changing partial text while dictating.

Parakeet Unified currently uses buffered streaming, so the left context is recomputed as the stream advances. That makes on-device real-time factor the key feasibility measurement: streaming only removes the final wait if the phone can sustain an RTF below 1.0 over a normal dictation session.

## SwiftKey

Select this app as Android's speech-recognition provider. In SwiftKey, disable **Multi-modal voice typing** if SwiftKey otherwise forces its own voice provider. Tapping the microphone can then invoke the system `RecognitionService` while SwiftKey remains the active keyboard.

## Performance logging

Logcat emits a line similar to:

```
buffered streaming: 30.0s audio, feed compute 12.00s (RTF 0.40), finalize 0.35s, wall 30.40s, buffered 0ms
```

The important values for phone testing are:

- **feed RTF < 1.0**: inference can keep up with live speech.
- **finalize time**: approximates the wait after speech ends.
- **wall time**: useful for diagnosing a stream that falls behind despite acceptable individual feeds.

Feed/finalize failures retain the session PCM and fall back to the existing one-shot transcription path. A stream-creation failure is surfaced as a recognition error rather than silently discarding the request.
