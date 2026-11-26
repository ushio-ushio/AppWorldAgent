"""Offline, zero-network diagnostics for cache_first trace snapshots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def analyze_trace(source: str | Path | list[dict[str, Any]]) -> dict[str, Any]:
    rows = source if isinstance(source, list) else [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    views = [row for row in rows if row.get("event") == "context_view" and isinstance(row.get("cache_first"), dict)]
    pairs=[]
    for left,right in zip(views,views[1:]):
        a=left.get("replay_snapshot",{}); b=right.get("replay_snapshot",{})
        x,y=a.get("serialized"),b.get("serialized")
        if isinstance(x,str) and isinstance(y,str):
            i=0
            while i<min(len(x),len(y)) and x[i]==y[i]: i+=1
            la=a.get("layers",[]); lb=b.get("layers",[]); layer="unknown"
            for ra,rb in zip(la,lb):
                if ra.get("sha256")!=rb.get("sha256"):
                    layer=ra.get("name","unknown"); break
            lcp=i
        else:
            i=right["cache_first"].get("lcp_chars",0); lcp=i; layer="unavailable_without_snapshot"
            la=left["cache_first"].get("replay_layers",[]); lb=right["cache_first"].get("replay_layers",[])
            for ra,rb in zip(la,lb):
                if ra.get("sha256")!=rb.get("sha256"):
                    layer=ra.get("name","unknown"); break
        m=right["cache_first"]
        pairs.append({"from_step":left.get("step"),"to_step":right.get("step"),
                      "lcp_chars":lcp,"lcp_ratio":m.get("lcp_ratio",0),
                      "first_difference_layer":layer,"first_difference_offset":i if x is not None else None,
                      "cache_efficiency":m.get("cached_tokens",0)/max(1,lcp//4)})
    return {"steps":len(views),"adjacent_steps":pairs,
            "compact_counts":{str(level):sum(v["cache_first"].get("compact_level")==level for v in views) for level in range(4)},
            "overflow_recoveries":sum(r.get("event")=="overflow_recovery" for r in rows),
            "clone":(views[-1].get("cache_first",{}).get("extraction",{}) if views else {})}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("trace_path",type=Path)
    args=parser.parse_args(); print(json.dumps(analyze_trace(args.trace_path),ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
