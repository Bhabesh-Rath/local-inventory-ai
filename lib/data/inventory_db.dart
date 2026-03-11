import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite3_flutter_libs/sqlite3_flutter_libs.dart';

/// File-backed SQLite (no codegen). Schema matches Places / Scans / DetectedItems.
class InventoryDb {
  InventoryDb._(this._db);

  final Database _db;

  static Future<InventoryDb> open() async {
    await applyWorkaroundToOpenSqlite3OnOldAndroidVersions();
    final dir = await getApplicationDocumentsDirectory();
    final file = File(p.join(dir.path, 'wheres_my_stuff.sqlite'));
    final db = sqlite3.open(file.path);
    db.execute('''
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS places (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  display_order INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  place_id INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
  timestamp_millis INTEGER NOT NULL,
  location_title TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detected_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_scans_place_time ON scans(place_id, timestamp_millis DESC);
CREATE INDEX IF NOT EXISTS idx_detected_scan ON detected_items(scan_id);
''');
    return InventoryDb._(db);
  }

  void close() => _db.dispose();

  int createPlace(String name, {int displayOrder = 0}) {
    _db.execute(
      'INSERT INTO places(name, display_order) VALUES(?, ?)',
      [name, displayOrder],
    );
    final r = _db.select('SELECT last_insert_rowid() AS id');
    return r.first['id'] as int;
  }

  void updatePlaceName(int placeId, String name) {
    _db.execute('UPDATE places SET name = ? WHERE id = ?', [name, placeId]);
  }

  void deletePlace(int placeId) {
    _db.execute('DELETE FROM places WHERE id = ?', [placeId]);
  }

  /// Latest scan for place, if any.
  ({int id, int timestampMillis, String locationTitle})? latestScan(int placeId) {
    final r = _db.select(
      'SELECT id, timestamp_millis, location_title FROM scans WHERE place_id = ? ORDER BY timestamp_millis DESC LIMIT 1',
      [placeId],
    );
    if (r.isEmpty) return null;
    final row = r.first;
    return (
      id: row['id'] as int,
      timestampMillis: row['timestamp_millis'] as int,
      locationTitle: row['location_title'] as String,
    );
  }

  /// Sum of counts from detected_items for the latest scan of [placeId].
  int itemCountForPlace(int placeId) {
    final scan = latestScan(placeId);
    if (scan == null) return 0;
    final r = _db.select(
      'SELECT COALESCE(SUM(count), 0) AS c FROM detected_items WHERE scan_id = ?',
      [scan.id],
    );
    return (r.first['c'] as int?) ?? 0;
  }

  /// Aggregated label -> total count for latest scan of place.
  Map<String, int> itemsForPlaceLatest(int placeId) {
    final scan = latestScan(placeId);
    if (scan == null) return {};
    final rows = _db.select(
      'SELECT label, SUM(count) AS c FROM detected_items WHERE scan_id = ? GROUP BY label ORDER BY label',
      [scan.id],
    );
    final m = <String, int>{};
    for (final row in rows) {
      m[row['label'] as String] = row['c'] as int;
    }
    return m;
  }

  int createScan({
    required int placeId,
    required String locationTitle,
  }) {
    final now = DateTime.now().millisecondsSinceEpoch;
    _db.execute(
      'INSERT INTO scans(place_id, timestamp_millis, location_title) VALUES(?, ?, ?)',
      [placeId, now, locationTitle],
    );
    final r = _db.select('SELECT last_insert_rowid() AS id');
    return r.first['id'] as int;
  }

  void addDetectedItem({
    required int scanId,
    required String label,
    required int count,
  }) {
    _db.execute(
      'INSERT INTO detected_items(scan_id, label, count) VALUES(?, ?, ?)',
      [scanId, label, count],
    );
  }

  /// Replace items for a scan with aggregated map (label -> count).
  void setDetectedItemsForScan(int scanId, Map<String, int> labelToCount) {
    _db.execute('DELETE FROM detected_items WHERE scan_id = ?', [scanId]);
    for (final e in labelToCount.entries) {
      if (e.value <= 0) continue;
      addDetectedItem(scanId: scanId, label: e.key, count: e.value);
    }
  }

  List<PlaceRow> allPlacesOrdered() {
    final rows = _db.select(
      'SELECT id, name, display_order FROM places ORDER BY display_order ASC, id ASC',
    );
    return rows
        .map((r) => PlaceRow(
              id: r['id'] as int,
              name: r['name'] as String,
              displayOrder: r['display_order'] as int,
            ))
        .toList();
  }

  /// "[object] last seen in [place] at [time]" or not found message.
  String searchInventory(String query) {
    final q = query.trim();
    if (q.isEmpty) return 'This item is not in inventory yet.';
    final rows = _db.select('''
SELECT di.label, p.name AS place_name, s.timestamp_millis
FROM detected_items di
JOIN scans s ON s.id = di.scan_id
JOIN places p ON p.id = s.place_id
WHERE di.label LIKE ?
ORDER BY s.timestamp_millis DESC
LIMIT 1
''', ['%$q%']);
    if (rows.isEmpty) return 'This item is not in inventory yet.';
    final row = rows.first;
    final label = row['label'] as String;
    final placeName = row['place_name'] as String;
    final ts = DateTime.fromMillisecondsSinceEpoch(row['timestamp_millis'] as int);
    final timeLabel =
        '${ts.year.toString().padLeft(4, '0')}-${ts.month.toString().padLeft(2, '0')}-${ts.day.toString().padLeft(2, '0')} '
        '${ts.hour.toString().padLeft(2, '0')}:${ts.minute.toString().padLeft(2, '0')}';
    return '$label last seen in $placeName at $timeLabel';
  }
}

class PlaceRow {
  PlaceRow({
    required this.id,
    required this.name,
    required this.displayOrder,
  });
  final int id;
  final String name;
  final int displayOrder;
}
