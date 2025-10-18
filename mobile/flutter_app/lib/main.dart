import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:image/image.dart' as img_pkg;
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:flutter/foundation.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Load environment variables from mobile/flutter_app/.env
  try {
    await dotenv.load(fileName: ".env");
  } catch (e) {
    // Ignore missing .env for web or test environments; we'll use defaults.
    // Print for developer visibility in debug builds.
    if (kDebugMode) {
      // ignore: avoid_print
      print('dotenv.load() failed: $e');
    }
  }
  runApp(const PotatoApp());
}

class PotatoApp extends StatelessWidget {
  const PotatoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Potato Leaf Detector',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: const HomePage(),
    );
  }
}

class Prediction {
  final String label;
  final double confidence;
  Prediction({required this.label, required this.confidence});
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker _picker = ImagePicker();
  XFile? _imageFile;
  Uint8List? _imageBytes;
  bool _loading = false;
  String? _error;
  List<Prediction>? _predictions;

  // Read endpoint and token from env (set .env file in mobile/flutter_app/)
  String get hfEndpoint {
    try {
      final runtime = dotenv.env['HF_RUNTIME'];
      if (runtime != null && runtime.isNotEmpty) {
        return '$runtime/api/predict';
      }
    } catch (_) {}
    return 'https://khalil-amamri-potato-space.hf.space/api/predict';
  }

  String? get hfToken {
    try {
      return dotenv.env['HF_TOKEN'];
    } catch (_) {
      return null;
    }
  }

  Map<String, String> get extraHeaders {
    final token = hfToken;
    if (token != null && token.isNotEmpty) {
      return {'Authorization': 'Bearer $token'};
    }
    return {};
  }

  Future<void> _pickFromCamera() async {
    final XFile? file =
        await _picker.pickImage(source: ImageSource.camera, imageQuality: 85);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _imageFile = file;
      _imageBytes = bytes;
      _predictions = null;
      _error = null;
    });
  }

  Future<void> _pickFromGallery() async {
    final XFile? file =
        await _picker.pickImage(source: ImageSource.gallery, imageQuality: 85);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _imageFile = file;
      _imageBytes = bytes;
      _predictions = null;
      _error = null;
    });
  }

  // Resize the image to max dim (keeps aspect ratio)
  Future<Uint8List> _resizeImage(Uint8List bytes, {int maxDim = 800}) async {
    final img_pkg.Image? image = img_pkg.decodeImage(bytes);
    if (image == null) return bytes;
    final int w = image.width;
    final int h = image.height;
    if (w <= maxDim && h <= maxDim) return bytes;
    final img_pkg.Image resized = img_pkg.copyResize(
      image,
      width: (w > h ? maxDim : (w * maxDim ~/ h)),
    );
    return Uint8List.fromList(img_pkg.encodeJpg(resized, quality: 85));
  }

  Future<void> _sendToModel() async {
    if (_imageFile == null) {
      setState(() {
        _error = 'Please take or select a photo first.';
      });
      return;
    }
    setState(() {
      _loading = true;
      _error = null;
      _predictions = null;
    });

    try {
      final Uint8List originalBytes =
          _imageBytes ?? await _imageFile!.readAsBytes();
      final Uint8List resizedBytes =
          await _resizeImage(originalBytes, maxDim: 600);
      final String base64Data = base64Encode(resizedBytes);
      final String dataUri = 'data:image/jpeg;base64,$base64Data';

      final body = jsonEncode({
        'data': [dataUri]
      });

      final headers = {
        'Content-Type': 'application/json',
        ...extraHeaders,
      };

      final resp = await http
          .post(
            Uri.parse(hfEndpoint),
            headers: headers,
            body: body,
          )
          .timeout(const Duration(seconds: 30));

      if (resp.statusCode != 200) {
        setState(() {
          _error = 'Server returned status ${resp.statusCode}';
          _loading = false;
        });
        return;
      }

      final Map<String, dynamic> jsonResp =
          jsonDecode(resp.body) as Map<String, dynamic>;

      List<Prediction> preds = [];
      if (jsonResp.containsKey('label')) {
        final label = jsonResp['label'].toString();
        double conf = 0.0;
        if (jsonResp['confidences'] is List) {
          final top = (jsonResp['confidences'] as List).firstWhere(
            (e) => e is Map && e['label'] == label,
            orElse: () => null,
          );
          if (top != null && top is Map && top['confidence'] != null) {
            conf = (top['confidence'] as num).toDouble();
          }
        }
        preds.add(Prediction(label: label, confidence: conf));
      } else if (jsonResp.containsKey('data')) {
        final data = jsonResp['data'];
        if (data is List && data.isNotEmpty) {
          final first = data[0];
          if (first is String) {
            preds.add(Prediction(label: first, confidence: 0.0));
          } else if (first is Map && first.containsKey('label')) {
            preds.add(Prediction(
              label: first['label'].toString(),
              confidence: (first['confidence'] ?? 0.0).toDouble(),
            ));
          }
        }
      } else {
        final text = resp.body.trim();
        preds.add(Prediction(label: text, confidence: 0.0));
      }

      setState(() {
        _predictions = preds;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Request failed: $e';
        _loading = false;
      });
    }
  }

  Widget _resultCard() {
    if (_loading) {
      return const Center(
          child: SpinKitThreeBounce(color: Colors.green, size: 36.0));
    }
    if (_error != null) {
      return Text(_error!, style: const TextStyle(color: Colors.red));
    }
    if (_predictions == null) {
      return const SizedBox.shrink();
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: _predictions!.map((p) {
        return Card(
          child: ListTile(
            title: Text(p.label,
                style: const TextStyle(fontWeight: FontWeight.bold)),
            subtitle:
                Text('Confidence: ${(p.confidence * 100).toStringAsFixed(1)}%'),
          ),
        );
      }).toList(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final preview = _imageFile == null || _imageBytes == null
        ? Container(
            height: 240,
            color: Colors.grey[200],
            child: const Center(
                child: Text('No image selected',
                    style: TextStyle(color: Colors.black54))),
          )
        : Image.memory(_imageBytes!, height: 240, fit: BoxFit.cover);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Potato Leaf Detector'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            preview,
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  onPressed: _pickFromCamera,
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.photo),
                  label: const Text('Gallery'),
                  onPressed: _pickFromGallery,
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.send),
                  label: const Text('Send'),
                  onPressed: _loading ? null : _sendToModel,
                ),
              ],
            ),
            const SizedBox(height: 18),
            _resultCard(),
            const SizedBox(height: 24),
            const Text(
              'Tips: Ensure the leaf fills most of the frame; good light; take multiple photos for better results.',
              style: TextStyle(color: Colors.black54),
            )
          ],
        ),
      ),
    );
  }
}
