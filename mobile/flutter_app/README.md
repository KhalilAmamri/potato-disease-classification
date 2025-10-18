# Potato Flutter App

Minimal Flutter app to capture a potato leaf photo and send it to your Hugging Face Space for prediction.

## Quick start

1. Make sure Flutter is installed and `flutter doctor` passes.
2. Open terminal in `mobile/flutter_app` and run:

```powershell
flutter pub get
flutter run
```

3. Replace the placeholder endpoint in `lib/main.dart`:

```dart
final String hfEndpoint = 'https://<YOUR-RUNTIME>.hf.space/api/predict';
```

If your Space is private, add the Authorization header in `extraHeaders`.

## Notes

- Android: add camera and storage permissions in `android/app/src/main/AndroidManifest.xml`.
- iOS: add NSCameraUsageDescription & NSPhotoLibraryUsageDescription to `ios/Runner/Info.plist`.
- The app resizes images before uploading to reduce bandwidth.
