"""Review completed E0 and freeze proposed criteria before any final training."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml

from underwater_vision.matchability.protocol import digest, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("outputs/matchability_labels_v3"))
    p.add_argument("--criteria", type=Path, default=Path("configs/matchability/criteria_v3.yaml"))
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    audit = read_json(a.dataset / "audit_v3.json")
    if audit["status"] != "complete":
        raise ValueError("E0 must be complete before criteria freeze")
    for name, report in audit["audits"].items():
        print(name, report["counts"]["pos_1.5_neg_4"], flush=True)
    if a.freeze:
        criteria = yaml.safe_load(a.criteria.read_text())
        if criteria["status"] == "frozen_after_E0" and criteria["label_audit_sha256"] == digest(
            a.dataset / "audit_v3.json"
        ):
            print("Existing frozen criteria match reproduced E0; criteria were not changed")
            return
        if criteria["status"] != "proposed_before_E0":
            raise ValueError(
                "Different E0 requires a new registered protocol; frozen criteria stay unchanged"
            )
        criteria.update(
            status="frozen_after_E0",
            frozen_at_utc=datetime.now(UTC).isoformat(),
            label_audit_sha256=digest(a.dataset / "audit_v3.json"),
            threshold_decision="Keep proposed 1.5/4 px, n_min=2, eligible minimum=15; threshold sensitivity reported in E0",
        )
        a.criteria.write_text(yaml.safe_dump(criteria, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
