import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/inventory_db.dart';
import '../providers/db_provider.dart';

class InventoryListTab extends ConsumerStatefulWidget {
  const InventoryListTab({super.key});

  @override
  ConsumerState<InventoryListTab> createState() => _InventoryListTabState();
}

class _InventoryListTabState extends ConsumerState<InventoryListTab> {
  String? _searchResult;

  @override
  Widget build(BuildContext context) {
    final dbAsync = ref.watch(inventoryDbProvider);
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      body: dbAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('DB error: $e')),
        data: (db) => CustomScrollView(
          slivers: [
            SliverAppBar(
              floating: true,
              pinned: true,
              title: const Text('Inventory'),
              bottom: PreferredSize(
                preferredSize: const Size.fromHeight(72),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
                  child: SearchBar(
                    hintText: 'Search your stuff…',
                    leading: const Icon(Icons.search),
                    padding: const MaterialStatePropertyAll(
                      EdgeInsets.symmetric(horizontal: 12),
                    ),
                    onChanged: (_) {},
                    onSubmitted: (q) async {
                      final r = db.searchInventory(q);
                      setState(() => _searchResult = r);
                    },
                    trailing: [
                      IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => setState(() {
                          _searchResult = null;
                        }),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            if (_searchResult != null)
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                  child: Card(
                    surfaceTintColor: cs.primary,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: Text(_searchResult!),
                    ),
                  ),
                ),
              ),
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                child: FilledButton.tonalIcon(
                  onPressed: () => _addPlace(context, db),
                  icon: const Icon(Icons.add_location_alt_outlined),
                  label: const Text('Add place'),
                ),
              ),
            ),
            StreamBuilder<List<PlaceRow>>(
              stream: _placesStream(db),
              builder: (context, snapshot) {
                final places = snapshot.data ?? [];
                if (places.isEmpty) {
                  return SliverFillRemaining(
                    hasScrollBody: false,
                    child: Center(
                      child: Text(
                        'No places yet.\nTap "Add place" to create a box.',
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.bodyLarge,
                      ),
                    ),
                  );
                }
                return SliverPadding(
                  padding: const EdgeInsets.fromLTRB(16, 0, 16, 100),
                  sliver: SliverList(
                    delegate: SliverChildBuilderDelegate(
                      (context, index) {
                        final place = places[index];
                        return _PlaceCard(
                          place: place,
                          db: db,
                          onCompareUpdate: () {
                            ref.read(scanContextProvider.notifier).state =
                                ScanContext(
                              placeId: place.id,
                              placeName: place.name,
                            );
                            // Parent HomeShell must switch tab - use callback via provider or global key.
                            // We'll use a simple provider for tab index.
                            ref.read(mainTabIndexProvider.notifier).state = 1;
                          },
                        );
                      },
                      childCount: places.length,
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Stream<List<PlaceRow>> _placesStream(InventoryDb db) async* {
    while (true) {
      yield db.allPlacesOrdered();
      await Future<void>.delayed(const Duration(milliseconds: 500));
    }
  }

  Future<void> _addPlace(BuildContext context, InventoryDb db) async {
    final c = TextEditingController();
    final name = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('New place'),
        content: TextField(
          controller: c,
          decoration: const InputDecoration(
            labelText: 'Place name',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
          onSubmitted: (v) => Navigator.pop(ctx, v.trim()),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, c.text.trim()),
            child: const Text('Add'),
          ),
        ],
      ),
    );
    if (name != null && name.isNotEmpty) {
      db.createPlace(name);
      setState(() {});
    }
  }
}

class _PlaceCard extends StatefulWidget {
  const _PlaceCard({
    required this.place,
    required this.db,
    required this.onCompareUpdate,
  });
  final PlaceRow place;
  final InventoryDb db;
  final VoidCallback onCompareUpdate;

  @override
  State<_PlaceCard> createState() => _PlaceCardState();
}

class _PlaceCardState extends State<_PlaceCard> {
  bool _expanded = false;
  late TextEditingController _nameController;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: widget.place.name);
  }

  @override
  void didUpdateWidget(covariant _PlaceCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.place.name != widget.place.name) {
      _nameController.text = widget.place.name;
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final count = widget.db.itemCountForPlace(widget.place.id);
    final items = widget.db.itemsForPlaceLatest(widget.place.id);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      surfaceTintColor: cs.primary,
      child: InkWell(
        onTap: () => setState(() => _expanded = !_expanded),
        borderRadius: BorderRadius.circular(20),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _nameController,
                      style: Theme.of(context).textTheme.titleMedium,
                      decoration: const InputDecoration(
                        border: InputBorder.none,
                        isDense: true,
                        contentPadding: EdgeInsets.zero,
                      ),
                      onSubmitted: (v) {
                        if (v.trim().isNotEmpty) {
                          widget.db.updatePlaceName(widget.place.id, v.trim());
                        }
                      },
                    ),
                  ),
                  Text(
                    '$count items',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: cs.onSurfaceVariant,
                        ),
                  ),
                  PopupMenuButton<String>(
                    onSelected: (value) async {
                      if (value == 'delete') {
                        final ok = await showDialog<bool>(
                          context: context,
                          builder: (ctx) => AlertDialog(
                            title: const Text('Delete place?'),
                            content: const Text(
                              'This removes the place and its scan history.',
                            ),
                            actions: [
                              TextButton(
                                onPressed: () => Navigator.pop(ctx, false),
                                child: const Text('Cancel'),
                              ),
                              FilledButton(
                                onPressed: () => Navigator.pop(ctx, true),
                                child: const Text('Delete'),
                              ),
                            ],
                          ),
                        );
                        if (ok == true) {
                          widget.db.deletePlace(widget.place.id);
                          setState(() {});
                        }
                      } else if (value == 'compare') {
                        widget.onCompareUpdate();
                      }
                    },
                    itemBuilder: (context) => [
                      const PopupMenuItem(
                        value: 'compare',
                        child: ListTile(
                          leading: Icon(Icons.compare_arrows),
                          title: Text('Compare & update'),
                          contentPadding: EdgeInsets.zero,
                        ),
                      ),
                      const PopupMenuItem(
                        value: 'delete',
                        child: ListTile(
                          leading: Icon(Icons.delete_outline),
                          title: Text('Delete place'),
                          contentPadding: EdgeInsets.zero,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              if (_expanded && items.isNotEmpty) ...[
                const SizedBox(height: 8),
                const Divider(),
                ...items.entries.map(
                  (e) => Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(e.key),
                        Text('× ${e.value}'),
                      ],
                    ),
                  ),
                ),
              ] else if (_expanded)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    'No items yet — scan to add.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
