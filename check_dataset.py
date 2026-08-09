from data_management.dataset_factory import DatasetFactory
tr, v, te, _ = DatasetFactory.create_datasets(
    dataset_type="ptbxl", download=False, resolution="lr"
)
print(f"Train: {len(tr)}, Val: {len(v)}, Test: {len(te)}, TOTAL: {len(tr)+len(v)+len(te)}")
