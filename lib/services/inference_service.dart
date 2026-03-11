import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// Loads model bytes on main isolate once; runs inference in a worker isolate
/// so the UI thread doesn't freeze.
class InferenceService {
  InferenceService._();
  static final InferenceService instance = InferenceService._();

  Uint8List? _modelBytes;
  List<String>? _labels;

  Future<void> ensureModelLoaded({String assetPath = 'assets/finetuned_model.tflite'}) async {
    if (_modelBytes != null) return;
    final data = await rootBundle.load(assetPath);
    _modelBytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await _loadLabels();
  }

  Future<void> _loadLabels({String labelsPath = 'assets/coco_labels.txt'}) async {
    if (_labels != null) return;
    final data = await rootBundle.loadString(labelsPath);
    _labels = data.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
  }

  Uint8List get modelBytes {
    final b = _modelBytes;
    if (b == null) {
      throw StateError('Call ensureModelLoaded first');
    }
    return b;
  }

  /// Returns inferences with a confidence score of 25% or higher.
  Future<List<({int index, double score})>> runInferenceOnImageBytes(
    Uint8List imageBytes,
  ) async {
    final modelBytes = this.modelBytes;
    return Isolate.run(() => _runInferenceSync(modelBytes, imageBytes));
  }

  String labelForIndex(int index) {
    final l = _labels;
    if (l == null) {
      throw StateError("Labels not loaded yet. Call ensureModelLoaded first.");
    }
    return l[index];
  }
}

/// Runs in worker isolate (Dart 3 `Isolate.run`).
List<({int index, double score})> _runInferenceSync(
  Uint8List modelBytes,
  Uint8List imageBytes,
) {
  File? tmp;
  Interpreter? interpreter;
  try {
    tmp = File(
      '${Directory.systemTemp.path}/wms_model_${DateTime.now().microsecondsSinceEpoch}.tflite',
    );
    tmp.writeAsBytesSync(modelBytes, flush: true);
    // fromFile may be sync or async depending on version; use fromBuffer sync if available
    interpreter = Interpreter.fromBuffer(
      modelBytes,
      options: InterpreterOptions()..threads = 2,
    );
    final input = _preprocessImageNet(imageBytes);
    final output = List.filled(80, 0.0).reshape([1, 80]);
    interpreter.run(input, output);
    interpreter.close();
    interpreter = null;
    final logits = List<double>.from(output[0] as List);
    final indexed = <({int index, double score})>[];
    for (var i = 0; i < logits.length; i++) {
      indexed.add((index: i, score: logits[i]));
    }
    indexed.sort((a, b) => b.score.compareTo(a.score));
    return indexed.where((element) => element.score >= 0.25).toList();
  } catch (e, st) {
    interpreter?.close();
    try {
      if (tmp != null && tmp.existsSync()) tmp.deleteSync();
    } catch (_) {}
    // ignore: avoid_print
    print('Inference error: $e\n$st');
    return [];
  } finally {
    try {
      if (tmp != null && tmp.existsSync()) tmp.deleteSync();
    } catch (_) {}
  }
}

/// NHWC [1,224,224,3] float32 ImageNet normalization.
List<List<List<List<double>>>> _preprocessImageNet(Uint8List imageBytes) {
  final decoded = img.decodeImage(imageBytes);
  if (decoded == null) {
    throw StateError('Could not decode image');
  }
  final resized = img.copyResize(decoded, width: 224, height: 224);
  final input = List.generate(
    1,
    (_) => List.generate(
      224,
      (y) => List.generate(
        224,
        (x) {
          final p = resized.getPixel(x, y);
          final r = p.r / 255.0;
          final g = p.g / 255.0;
          final b = p.b / 255.0;
          return [
            (r - 0.485) / 0.229,
            (g - 0.456) / 0.224,
            (b - 0.406) / 0.225,
          ];
        },
      ),
    ),
  );
  return input;
}
