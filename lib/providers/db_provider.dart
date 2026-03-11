import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/inventory_db.dart';

final inventoryDbProvider = FutureProvider<InventoryDb>((ref) async {
  final db = await InventoryDb.open();
  ref.onDispose(db.close);
  return db;
});

/// When set, Scan tab is updating this place (compare & update flow).
final scanContextProvider = StateProvider<ScanContext?>((ref) => null);

class ScanContext {
  ScanContext({required this.placeId, required this.placeName});
  final int placeId;
  final String placeName;
}

/// Inventory = 0, Scan = 1. Used to switch tab from Inventory (compare & update).
final mainTabIndexProvider = StateProvider<int>((ref) => 0);
