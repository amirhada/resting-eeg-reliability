# dc_reliability.py

from __future__ import annotations

import os
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Tuple

import pandas as pd
import datachain as dc
from datachain import C, File, DataModel

import reliability_core as rc


# Paths to repo-local inputs (these live in the GitHub repo)
MIDS_EXCEL_PATH = Path("mids-list.xlsx")
GEOM_PATH = Path("your_headset_geometry.json")

# DataChain dataset names – adapt if your canonical names differ
CANONICAL_EEG_DATASET = os.environ.get(
    "CANONICAL_EEG_DATASET", "canonical.eeg.post-ica-eeg"
)
PARSED_EVENTS_DATASET = os.environ.get(
    "PARSED_EVENTS_DATASET", "canonical.events.parsed-events"
)


class ClientReliabilitySummary(DataModel):
    """
    Simple DataModel for saving per-client outputs in DataChain.
    """
    client_id: str
    summary_table: File
    detail_table: File
    conclusion_text: str


def load_mids_from_excel() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Read mids-list.xlsx and return:
      - all_go_mids: list of all MIDs marked 'Go'
      - client_to_mids: dict[client_id -> list of Go MIDs]
    """
    df = pd.read_excel(MIDS_EXCEL_PATH)

    go_df = df[df["No Go"].astype(str).str.strip().eq("Go")].copy()
    go_df["MID"] = go_df["MID"].astype(str)
    go_df["ID"] = go_df["ID"].astype(str)

    all_go_mids = sorted(go_df["MID"].unique().tolist())
    client_to_mids = go_df.groupby("ID")["MID"].apply(list).to_dict()
    return all_go_mids, client_to_mids


def get_base_session_dataset(all_go_mids: List[str]):
    """
    Build a DataChain dataset of EEG + events for only the Go MIDs.
    """
    chain = (
        dc.read_dataset(CANONICAL_EEG_DATASET, update=True)
        .merge(
            dc.read_dataset(PARSED_EVENTS_DATASET, update=True),
            on="measurement_id",
            inner=True,
        )
        .filter(C("measurement_id").in_(all_go_mids))
        .select("measurement_id", "eeg_data", "events_file")
        .persist()
    )
    return chain


def run_query():
    """
    Main entrypoint for DataChain / Datapoint.

    This function is what you point Studio at.
    """
    # 1) Excel → mids
    all_go_mids, client_to_mids = load_mids_from_excel()

    # 2) DataChain EEG+events dataset for those MIDs
    sessions_chain = get_base_session_dataset(all_go_mids)

    # 3) Geometry shared across clients
    geom_json = GEOM_PATH.read_text(encoding="utf-8")

    # 4) Loop over clients in Python
    results: List[Tuple[str, File, File, str]] = []

    for client_id, mids in client_to_mids.items():
        # Filter rows for this client's sessions
        client_rows = (
            sessions_chain
            .filter(C("measurement_id").in_(mids))
            .select("measurement_id", "eeg_data", "events_file")
        )

        # Materialize rows to pandas.
        # If your DataChain version uses a different method, replace `.to_pandas()`.
        try:
            rows = client_rows.to_pandas()
        except AttributeError:
            raise RuntimeError(
                "Replace `client_rows.to_pandas()` with the correct "
                "DataChain materialization method for your environment."
            )

        if rows.empty:
            continue

        # 5) Per-client temp directory
        tmp_root = Path(os.environ.get("DC_LOCAL_TMP", "/tmp"))
        client_dir = tmp_root / f"client_{client_id}"
        client_dir.mkdir(parents=True, exist_ok=True)

        # Write geometry file
        geom_path = client_dir / "your_headset_geometry.json"
        geom_path.write_text(geom_json, encoding="utf-8")

        # Stage EEG + events with the naming the core expects
        for _, r in rows.iterrows():
            mid = str(r["measurement_id"])
            eeg_file: File = r["eeg_data"]
            ev_file: File = r["events_file"]

            eeg_local = Path(eeg_file.get_local_path())
            ev_local = Path(ev_file.get_local_path())

            (client_dir / f"{mid}.parquet").write_bytes(eeg_local.read_bytes())
            (client_dir / f"{mid}e.parquet").write_bytes(ev_local.read_bytes())

        # 6) Core pipeline
        rec_df = rc.build_rec_df_for_client(
            client_id=client_id,
            mids=mids,
            data_dir=str(client_dir),
            json_path=str(geom_path),
        )
        if rec_df is None or rec_df.empty:
            continue

        client_summary, client_detail = rc.compute_reliability_for_client(
            client_id, rec_df
        )
        conclusion_text, _ = rc.summarize_client_conclusions(
            client_id, client_summary
        )

        # 7) Upload summary + detail as CSV to S3 via DataChain File API
        def upload_df(df: pd.DataFrame, name: str) -> File:
            buf = BytesIO()
            df.to_csv(buf, index=False)
            path = f"reliability/{client_id}/{name}.csv"
            return File.upload(buf.getvalue(), path)

        summary_file = upload_df(client_summary, "summary")
        detail_file = upload_df(client_detail, "detail")

        results.append((client_id, summary_file, detail_file, conclusion_text))

    # 8) Optional: convert results to a DataChain dataset
    # If you want a dataset out of this, you can do:
    #
    # out_df = pd.DataFrame([
    #     {
    #         "client_id": cid,
    #         "summary_table": s,
    #         "detail_table": d,
    #         "conclusion_text": txt,
    #     }
    #     for cid, s, d, txt in results
    # ])
    # dc.from_pandas(out_df, ClientReliabilitySummary).save("your.output.dataset")
    #
    # For now we just return the list of tuples.

    return results


if __name__ == "__main__":
    run_query()
