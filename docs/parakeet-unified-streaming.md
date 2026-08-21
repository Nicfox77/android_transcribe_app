# Parakeet Unified streaming voice input

This fork adds true incremental inference to Android's `RecognitionService` when the active transcribe.cpp model advertises the Parakeet buffered-streaming extension.

## Recommended model

Use `handy-computer/parakeet-unified-en-0.6b-gguf` with:

- File: `parakeet-unified-en-0.6b-Q4_K_M.gguf`
- Size: about 477 MB
- SHA-256: `a8bf3de2b393bd14ead5a858c3748d5e3b07a20fdeabdd3b498fba4f463fa929`
- Download: `https://huggingface.co/handy-computer/parakeet-unified-en-0.6b-gguf/resolve/main/parakeet-unified-en-0.6b-Q4_K_M.gguf`

Download the GGUF on Android, open **Models** in Offline Voice Input, import it, and select it as the active model.

The built-in Parakeet TDT v3 model is intentionally retained as the fallback model. It is offline-only and therefore continues to use the existing batch transcription path.

## Streaming profile

The RecognitionService uses the 1.12-second Parakeet Unified profile:

- left context: 5600 ms
- chunk: 560 ms
- right context: 560 ms
- commit policy: `OnFinalize`

Audio is fed to the model continuously while the microphone is active, but Android receives only the final stable transcript. This is meant to remove most of the post-recording wait without showing changing partial text while dictating.

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

If native streaming fails, the request automatically falls back to the existing one-shot transcription path using the audio retained for that recognition session.
