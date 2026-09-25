from er945.split import make_split, _fold


def test_split_is_deterministic():
    gt = {f"S1-{i:05d}": set() for i in range(100)}
    t1, v1 = make_split(gt)
    t2, v2 = make_split(gt)
    assert t1.keys() == t2.keys()
    assert v1.keys() == v2.keys()


def test_split_is_disjoint_and_complete():
    gt = {f"S1-{i:05d}": set() for i in range(200)}
    train_gt, val_gt = make_split(gt)
    assert train_gt.keys().isdisjoint(val_gt.keys())
    assert len(train_gt) + len(val_gt) == len(gt)


def test_split_ratio_roughly_80_20():
    gt = {f"S1-{i:05d}": set() for i in range(1000)}
    _, val_gt = make_split(gt)
    ratio = len(val_gt) / 1000
    assert 0.15 <= ratio <= 0.25, f"Unexpected val ratio: {ratio}"


def test_no_s1_id_in_matches():
    """S1 IDs must never appear in matched_entity_ids."""
    gt = {"S1-00001": {"S2-00001", "S3-00001"}, "S1-00002": set()}
    train_gt, val_gt = make_split(gt)
    for matches in {**train_gt, **val_gt}.values():
        assert all(not m.startswith("S1-") for m in matches)
