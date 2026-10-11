"""Open the local audit and SAM-assisted hand-mask editor (no external upload)."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default="outputs/semantic_labels_v2")
    p.add_argument("--audit", default="outputs/semantic_audit_v2")
    p.add_argument("--port", type=int, default=8010)
    a = p.parse_args()
    import uvicorn

    from underwater_vision.serving.review import create_review_app

    uvicorn.run(create_review_app(a.labels, a.audit), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
