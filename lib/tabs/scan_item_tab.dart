import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/inventory_db.dart';
import '../providers/db_provider.dart';
import '../services/inference_service.dart';

class ScanItemTab extends ConsumerStatefulWidget {
  const ScanItemTab({super.key});

  @override
  ConsumerState<ScanItemTab> createState() => _ScanItemTabState();
}

class _ScanItemTabState extends ConsumerState<ScanItemTab> {
  CameraController? _controller;
  Object? _cameraError;
  bool _loading = true;
  bool _capturing = false;

  /// After capture: show still image + dots.
  Uint8List? _capturedJpeg;
  final List<String> _selectedLabels = [];
  int? _pendingScanId;

  @override
  void initState() {
    super.initState();
    _initCamera();
    InferenceService.instance.ensureModelLoaded();
  }

  Future<void> _initCamera() async {
    setState(() {
      _loading = true;
      _cameraError = null;
    });
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw StateError('No available cameras.');
      }
      final back = cameras
          .where((c) => c.lensDirection == CameraLensDirection.back)
          .toList();
      final camera = back.isNotEmpty
          ? back.first
          : (cameras.isNotEmpty ? cameras.first : null);
      if (camera == null) throw StateError('No cameras found.');
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      _controller = controller;
    } catch (e) {
      _cameraError = e;
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    setState(() => _capturing = true);
    try {
      final file = await controller.takePicture();
      final bytes = await File(file.path).readAsBytes();
      if (!mounted) return;
      setState(() {
        _capturedJpeg = bytes;
        _selectedLabels.clear();
        _pendingScanId = null;
      });
      await _runInference();
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  Future<void> _runInference() async {
    final jpeg = _capturedJpeg;
    if (jpeg == null) return;
    try {
      final results = await InferenceService.instance.runInferenceOnImageBytes(jpeg);
      if (!mounted) return;
      _showLabelDialog(results);
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Inference failed')),
        );
      }
    }
  }

  Future<void> _showLabelDialog(List<({int index, double score})> results) async {
    final labels = results.map((e) => InferenceService.instance.labelForIndex(e.index)).toList();
    final selected = <String>{};

    await showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Select labels'),
        content: StatefulBuilder(
          builder: (context, setState) {
            return Column(
              mainAxisSize: MainAxisSize.min,
              children: labels.map((label) {
                return CheckboxListTile(
                  title: Text(label),
                  value: selected.contains(label),
                  onChanged: (value) {
                    setState(() {
                      if (value == true) {
                        selected.add(label);
                      } else {
                        selected.remove(label);
                      }
                    });
                  },
                );
              }).toList(),
            );
          },
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              setState(() {
                _capturedJpeg = null;
                _selectedLabels.clear();
              });
            },
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.of(context).pop();
              if (selected.isNotEmpty) {
                _promptForPlaceNameAndSave(selected.toList());
              } else {
                setState(() {
                  _capturedJpeg = null;
                  _selectedLabels.clear();
                });
              }
            },
            child: const Text('Done'),
          ),
        ],
      ),
    );
  }

  Future<void> _promptForPlaceNameAndSave(List<String> labels) async {
    final ctx = ref.read(scanContextProvider);
    String? placeName = ctx?.placeName;

    if (placeName == null) {
      final c = TextEditingController();
      placeName = await showDialog<String>(
        context: context,
        builder: (dCtx) => AlertDialog(
          title: const Text('Save to place'),
          content: TextField(
            controller: c,
            decoration: const InputDecoration(
              labelText: 'Place name',
              border: OutlineInputBorder(),
            ),
            autofocus: true,
            onSubmitted: (v) => Navigator.pop(dCtx, v.trim()),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dCtx),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(dCtx, c.text.trim()),
              child: const Text('Save'),
            ),
          ],
        ),
      );
    }

    if (placeName != null && placeName.isNotEmpty) {
      await _saveSelection(labels, placeName);
    } else {
      // User cancelled or entered empty name, go back to camera view.
      if (mounted) {
        setState(() {
          _capturedJpeg = null;
          _selectedLabels.clear();
          _pendingScanId = null;
        });
      }
    }
  }

  Future<void> _saveSelection(List<String> labels, String placeName) async {
    final db = await ref.read(inventoryDbProvider.future);

    final allPlaces = db.allPlacesOrdered();
    PlaceRow? place;
    try {
      place = allPlaces.firstWhere((p) => p.name == placeName);
    } catch (e) {
      place = null;
    }

    final int placeId;
    if (place == null) {
      placeId = db.createPlace(placeName);
    } else {
      placeId = place.id;
    }

    _pendingScanId ??= db.createScan(
      placeId: placeId,
      locationTitle: placeName,
    );
    final scanId = _pendingScanId!;
    final existing = <String, int>{};
    for (final label in labels) {
      existing[label] = (existing[label] ?? 0) + 1;
    }
    db.setDetectedItemsForScan(scanId, existing);
    setState(() {
      _selectedLabels.addAll(labels);
    });
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Saved: ${labels.join(', ')} to $placeName')),
      );
      setState(() {
        _capturedJpeg = null;
        _selectedLabels.clear();
        _pendingScanId = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final ctx = ref.watch(scanContextProvider);
    final controller = _controller;

    if (_capturedJpeg != null) {
      return _buildReview(context, ctx);
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(ctx != null ? 'Scan: ${ctx.placeName}' : 'Scan'),
        actions: [
          IconButton(
            tooltip: 'Reload camera',
            onPressed: _loading ? null : _initCamera,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (ctx != null)
              Card(
                child: ListTile(
                  leading: const Icon(Icons.place_outlined),
                  title: Text('Updating "${ctx.placeName}"'),
                  subtitle: const Text(
                    'Capture replaces latest item list for this place when you save.',
                  ),
                ),
              ),
            const SizedBox(height: 8),
            Expanded(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: _buildPreview(controller),
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: _capturing ||
                      controller == null ||
                      !controller.value.isInitialized
                  ? null
                  : _takePicture,
              icon: _capturing
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.camera_alt),
              label: Text(_capturing ? 'Capturing…' : 'Take picture & scan'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreview(CameraController? controller) {
    if (_loading && controller == null) {
      return const Center(child: CircularProgressIndicator());
    }
    if (_cameraError != null) {
      return const Center(child: Text('Camera error: No available camera can be found'));
    }
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: Text('Camera not ready'));
    }
    return CameraPreview(controller);
  }

  Widget _buildReview(BuildContext context, ScanContext? ctx) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Label items'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => setState(() {
            _capturedJpeg = null;
            _selectedLabels.clear();
          }),
        ),
      ),
      body: Image.memory(
        _capturedJpeg!,
        fit: BoxFit.contain,
      ),
    );
  }
}
