import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelService {
  ModelService._();

  static final ModelService instance = ModelService._();

  Interpreter? _interpreter;

  bool get isLoaded => _interpreter != null;
  Interpreter get interpreter {
    final i = _interpreter;
    if (i == null) {
      throw StateError(
        'Model not loaded yet. Call ModelService.instance.load() first.',
      );
    }
    return i;
  }

  Future<void> load({
    String assetPath = 'assets/finetuned_model.tflite',
    int threads = 2,
  }) async {
    if (_interpreter != null) return;

    final data = await rootBundle.load(assetPath);
    final buffer = data.buffer;
    final bytes = buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);

    final options = InterpreterOptions()..threads = threads;

    _interpreter = await Interpreter.fromBuffer(
      Uint8List.fromList(bytes),
      options: options,
    );
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}

