# Standalone streaming IME validation

This branch validates the dedicated Offline Voice Input keyboard with Parakeet Unified EN 0.6B Q4_K_M buffered streaming.

The validation build uses the `(70, 7, 7)` profile (5600 ms left context, 560 ms chunk, 560 ms right context). Microphone PCM is fed to the native stream while recording. On Stop, the stream is finalized and its final snapshot is returned to the IME.

For robustness testing, the validation APK also checks the committed/tentative snapshot when `full` is empty. If all finalized stream text is empty, retained PCM is transcribed through the existing one-shot path instead of silently returning no text.
