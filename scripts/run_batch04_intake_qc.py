"""Run intake QC for real_export_batch_04 without building manifests or splits."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch_04"
RAW_ROOT = PROJECT_ROOT / "data/raw/real_export_batch_04"
OUTPUT_DIR = PROJECT_ROOT / "artifacts/reports/real_export_batch_04/intake_qc"
QC_REFERENCE_DATE = date(2026, 6, 9)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PATIENT_UID_PATTERN = re.compile(r"^patient_\d+$", re.IGNORECASE)
PATIENT_UID_PREFIX_PATTERN = re.compile(r"^(patient_\d+)(?:\b|_)", re.IGNORECASE)
OCT_FILENAME_PATTERN = re.compile(
    r"^(?P<exam_id>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<eye>[A-Za-z]+)",
    re.IGNORECASE,
)


@dataclass
class ImageRecord:
    patient_uid: str
    source_patient_folder: str
    path: Path
    modality: str
    exam_id: str
    exam_date: str
    exam_time: str
    eye: str
    can_parse_exam_date: bool
    can_parse_eye: bool
    date_is_future: bool


def resolve_patient_root(raw_root: Path) -> Path:
    direct_patient_dirs = [
        path for path in raw_root.iterdir() if path.is_dir() and PATIENT_UID_PREFIX_PATTERN.match(path.name)
    ]
    if direct_patient_dirs:
        return raw_root

    for container_name in ("patient", "patients"):
        container = raw_root / container_name
        if container.is_dir():
            nested_patient_dirs = [
                path for path in container.iterdir() if path.is_dir() and PATIENT_UID_PREFIX_PATTERN.match(path.name)
            ]
            if nested_patient_dirs:
                return container

    return raw_root


def normalize_eye(value: str) -> str:
    token = (value or "").strip().lower()
    if token in {"r", "right", "od"}:
        return "OD"
    if token in {"l", "left", "os"}:
        return "OS"
    return "unknown"


def normalize_date(value: str) -> tuple[str, bool, bool]:
    if len(value) != 8 or not value.isdigit():
        return "", False, False
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return "", False, False
    return parsed.isoformat(), True, parsed > QC_REFERENCE_DATE


def normalize_time(value: str) -> str:
    if len(value) == 6 and value.isdigit():
        return f"{value[:2]}:{value[2:4]}:{value[4:]}"
    return ""


def parse_oct_metadata(path: Path) -> dict[str, object]:
    match = OCT_FILENAME_PATTERN.match(path.stem)
    if not match:
        return {
            "exam_id": "",
            "exam_date": "",
            "exam_time": "",
            "eye": "unknown",
            "can_parse_exam_date": False,
            "can_parse_eye": False,
            "date_is_future": False,
        }

    groups = match.groupdict()
    exam_date, can_parse_exam_date, date_is_future = normalize_date(groups["date"])
    eye = normalize_eye(groups["eye"])
    return {
        "exam_id": groups["exam_id"],
        "exam_date": exam_date,
        "exam_time": normalize_time(groups["time"]),
        "eye": eye,
        "can_parse_exam_date": can_parse_exam_date,
        "can_parse_eye": eye in {"OD", "OS"},
        "date_is_future": date_is_future,
    }


def is_2d_analysis_path(path: Path) -> bool:
    joined = " ".join(part.lower() for part in path.parts)
    return any(keyword in joined for keyword in ("2danalysis", "2d_analysis", "2d analysis"))


def classify_image(path: Path) -> str:
    joined = " ".join(part.lower() for part in path.parts)
    if "ubm" in joined:
        return "ubm"
    if is_2d_analysis_path(path):
        return "oct_2d_analysis"
    if "oct" in joined or "casia" in path.stem.lower():
        return "oct_raw"
    return "other"


def assign_patient_uid(folder_name: str, index: int) -> str:
    if PATIENT_UID_PATTERN.match(folder_name):
        return folder_name.lower()
    prefix_match = PATIENT_UID_PREFIX_PATTERN.match(folder_name)
    if prefix_match:
        return prefix_match.group(1).lower()
    return f"patient_{index:03d}"


def safe_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def join_unique(values: Iterable[object]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"]
    return ";".join(dict.fromkeys(clean))


def load_prior_ids() -> tuple[set[str], set[str]]:
    patient_uids: set[str] = set()
    global_patient_uids: set[str] = set()
    paths = [
        PROJECT_ROOT / "data/manifests/real_export_batch_01_summary.csv",
        PROJECT_ROOT / "data/manifests/real_export_batch_02_summary.csv",
        PROJECT_ROOT / "data/manifests/real_export_batch_03_summary.csv",
        PROJECT_ROOT / "data/manifests/vault_as_oct_only_pod1_manifest_combined_batch_01_02_03_strict.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "patient_uid" in df.columns:
            patient_uids.update(df["patient_uid"].dropna().astype(str).str.lower())
        if "global_patient_uid" in df.columns:
            global_patient_uids.update(df["global_patient_uid"].dropna().astype(str))
    return patient_uids, global_patient_uids


def collect_records(patient_dirs: list[Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient_uid = assign_patient_uid(patient_dir.name, index)
        for path in sorted(patient_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            modality = classify_image(path)
            metadata = (
                parse_oct_metadata(path)
                if modality in {"oct_raw", "oct_2d_analysis"}
                else {
                    "exam_id": "",
                    "exam_date": "",
                    "exam_time": "",
                    "eye": "unknown",
                    "can_parse_exam_date": False,
                    "can_parse_eye": False,
                    "date_is_future": False,
                }
            )
            records.append(
                ImageRecord(
                    patient_uid=patient_uid,
                    source_patient_folder=patient_dir.name,
                    path=path,
                    modality=modality,
                    exam_id=str(metadata["exam_id"]),
                    exam_date=str(metadata["exam_date"]),
                    exam_time=str(metadata["exam_time"]),
                    eye=str(metadata["eye"]),
                    can_parse_exam_date=bool(metadata["can_parse_exam_date"]),
                    can_parse_eye=bool(metadata["can_parse_eye"]),
                    date_is_future=bool(metadata["date_is_future"]),
                )
            )
    return records


def build_visit_inventory(records: list[ImageRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[ImageRecord]] = defaultdict(list)
    for record in records:
        if record.modality in {"oct_raw", "oct_2d_analysis"}:
            grouped[(record.patient_uid, record.eye, record.exam_date)].append(record)

    by_eye: dict[tuple[str, str], list[tuple[str, list[ImageRecord]]]] = defaultdict(list)
    for (patient_uid, eye, exam_date), group in grouped.items():
        by_eye[(patient_uid, eye)].append((exam_date, group))

    for (patient_uid, eye), visits in sorted(by_eye.items()):
        visits.sort(key=lambda item: (item[0] or "9999-99-99", item[1][0].exam_time, safe_relative(item[1][0].path)))
        valid_dates = [exam_date for exam_date, _ in visits if exam_date]
        first_postop_date = valid_dates[1] if len(valid_dates) > 1 else ""
        for visit_index, (exam_date, group) in enumerate(visits):
            is_preop = bool(exam_date and valid_dates and exam_date == valid_dates[0])
            is_postop = bool(exam_date and first_postop_date and exam_date >= first_postop_date)
            postop_day = ""
            if is_postop:
                postop_day = (datetime.strptime(exam_date, "%Y-%m-%d") - datetime.strptime(first_postop_date, "%Y-%m-%d")).days + 1
            rows.append(
                {
                    "patient_uid": patient_uid,
                    "source_patient_folder": group[0].source_patient_folder,
                    "eye": eye,
                    "exam_date": exam_date,
                    "visit_index": visit_index,
                    "is_preop": is_preop,
                    "is_postop": is_postop,
                    "postop_day": postop_day,
                    "num_oct_raw_images": sum(record.modality == "oct_raw" for record in group),
                    "num_2d_analysis_images": sum(record.modality == "oct_2d_analysis" for record in group),
                    "num_exam_time_groups": len({(record.exam_id, record.exam_time) for record in group}),
                    "can_parse_exam_date": all(record.can_parse_exam_date for record in group),
                    "can_parse_eye": eye in {"OD", "OS"},
                    "date_is_future": any(record.date_is_future for record in group),
                    "exam_times": join_unique(record.exam_time for record in group),
                    "example_paths": join_unique(safe_relative(record.path) for record in group[:4]),
                }
            )

    columns = [
        "patient_uid",
        "source_patient_folder",
        "eye",
        "exam_date",
        "visit_index",
        "is_preop",
        "is_postop",
        "postop_day",
        "num_oct_raw_images",
        "num_2d_analysis_images",
        "num_exam_time_groups",
        "can_parse_exam_date",
        "can_parse_eye",
        "date_is_future",
        "exam_times",
        "example_paths",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["patient_uid", "eye", "visit_index"], kind="stable")


def build_patient_inventory(
    patient_dirs: list[Path],
    records: list[ImageRecord],
    visit_df: pd.DataFrame,
    prior_patient_uids: set[str],
    prior_global_patient_uids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    records_by_patient: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_patient[record.patient_uid].append(record)

    for index, patient_dir in enumerate(patient_dirs, start=1):
        patient_uid = assign_patient_uid(patient_dir.name, index)
        patient_records = records_by_patient.get(patient_uid, [])
        patient_visits = visit_df[visit_df["patient_uid"] == patient_uid] if not visit_df.empty else pd.DataFrame()
        eyes = sorted(set(patient_visits.loc[patient_visits["eye"].isin(["OD", "OS"]), "eye"])) if not patient_visits.empty else []
        preop_eyes = sorted(
            set(
                patient_visits.loc[
                    (patient_visits["eye"].isin(["OD", "OS"]))
                    & (patient_visits["is_preop"].astype(bool))
                    & (patient_visits["num_oct_raw_images"] > 0),
                    "eye",
                ]
            )
        ) if not patient_visits.empty else []
        pod1_eyes = sorted(
            set(
                patient_visits.loc[
                    (patient_visits["eye"].isin(["OD", "OS"]))
                    & (patient_visits["is_postop"].astype(bool))
                    & (patient_visits["postop_day"].astype(str) == "1")
                    & (patient_visits["num_2d_analysis_images"] > 0),
                    "eye",
                ]
            )
        ) if not patient_visits.empty else []
        global_patient_uid = f"{BATCH_ID}__{patient_uid}"
        rows.append(
            {
                "patient_uid": patient_uid,
                "global_patient_uid": global_patient_uid,
                "source_patient_folder": patient_dir.name,
                "is_strict_patient_folder_name": bool(PATIENT_UID_PATTERN.match(patient_dir.name)),
                "naming_issue": "" if PATIENT_UID_PATTERN.match(patient_dir.name) else "non_strict_patient_folder_name",
                "duplicates_prior_patient_uid": patient_uid in prior_patient_uids,
                "duplicates_prior_global_patient_uid": global_patient_uid in prior_global_patient_uids,
                "has_OD": "OD" in eyes,
                "has_OS": "OS" in eyes,
                "eyes_detected": join_unique(eyes),
                "num_eyes_detected": len(eyes),
                "eyes_with_preop_as_oct": join_unique(preop_eyes),
                "num_eyes_with_preop_as_oct": len(preop_eyes),
                "eyes_with_pod1_2d_analysis": join_unique(pod1_eyes),
                "num_eyes_with_pod1_2d_analysis": len(pod1_eyes),
                "num_all_images": len(patient_records),
                "num_oct_raw_images": sum(record.modality == "oct_raw" for record in patient_records),
                "num_2d_analysis_images": sum(record.modality == "oct_2d_analysis" for record in patient_records),
                "num_other_images": sum(record.modality == "other" for record in patient_records),
                "num_ubm_images": sum(record.modality == "ubm" for record in patient_records),
                "has_unparsed_oct_metadata": any(
                    record.modality in {"oct_raw", "oct_2d_analysis"}
                    and (not record.can_parse_exam_date or not record.can_parse_eye)
                    for record in patient_records
                ),
                "has_future_exam_date": any(record.date_is_future for record in patient_records),
            }
        )

    columns = [
        "patient_uid",
        "global_patient_uid",
        "source_patient_folder",
        "is_strict_patient_folder_name",
        "naming_issue",
        "duplicates_prior_patient_uid",
        "duplicates_prior_global_patient_uid",
        "has_OD",
        "has_OS",
        "eyes_detected",
        "num_eyes_detected",
        "eyes_with_preop_as_oct",
        "num_eyes_with_preop_as_oct",
        "eyes_with_pod1_2d_analysis",
        "num_eyes_with_pod1_2d_analysis",
        "num_all_images",
        "num_oct_raw_images",
        "num_2d_analysis_images",
        "num_other_images",
        "num_ubm_images",
        "has_unparsed_oct_metadata",
        "has_future_exam_date",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("patient_uid", kind="stable")


def add_issue(rows: list[dict[str, object]], patient_uid: str, folder: str, eye: str, issue_type: str, detail: str) -> None:
    rows.append(
        {
            "patient_uid": patient_uid,
            "source_patient_folder": folder,
            "eye": eye,
            "issue_type": issue_type,
            "detail": detail,
        }
    )


def build_missing_report(patient_df: pd.DataFrame, visit_df: pd.DataFrame, records: list[ImageRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in patient_df.to_dict(orient="records"):
        patient_uid = str(row["patient_uid"])
        folder = str(row["source_patient_folder"])
        if str(row["naming_issue"]):
            add_issue(rows, patient_uid, folder, "", "naming_anomaly", str(row["naming_issue"]))
        if bool(row["duplicates_prior_patient_uid"]):
            add_issue(rows, patient_uid, folder, "", "duplicate_patient_uid", "patient_uid exists in batch_01/batch_02/batch_03")
        if bool(row["duplicates_prior_global_patient_uid"]):
            add_issue(rows, patient_uid, folder, "", "duplicate_global_patient_uid", "global_patient_uid already exists")
        if not bool(row["has_OD"]):
            add_issue(rows, patient_uid, folder, "OD", "missing_eye", "OD not detected from OCT filenames")
        if not bool(row["has_OS"]):
            add_issue(rows, patient_uid, folder, "OS", "missing_eye", "OS not detected from OCT filenames")
        for eye in ["OD", "OS"]:
            eye_visits = visit_df[(visit_df["patient_uid"] == patient_uid) & (visit_df["eye"] == eye)] if not visit_df.empty else pd.DataFrame()
            if eye_visits.empty:
                continue
            preop = eye_visits[(eye_visits["is_preop"].astype(bool)) & (eye_visits["num_oct_raw_images"] > 0)]
            pod1 = eye_visits[
                (eye_visits["is_postop"].astype(bool))
                & (eye_visits["postop_day"].astype(str) == "1")
                & (eye_visits["num_2d_analysis_images"] > 0)
            ]
            if preop.empty:
                add_issue(rows, patient_uid, folder, eye, "missing_preop_as_oct", "no preoperative raw AS-OCT visit detected")
            if pod1.empty:
                add_issue(rows, patient_uid, folder, eye, "missing_pod1_2d_analysis", "no first-postoperative 2DAnalysis detected")
            for visit in eye_visits.to_dict(orient="records"):
                prefix = f"exam_date={visit['exam_date']}, visit_index={visit['visit_index']}"
                if not bool(visit["can_parse_exam_date"]):
                    add_issue(rows, patient_uid, folder, eye, "unparsed_exam_date", prefix)
                if not bool(visit["can_parse_eye"]):
                    add_issue(rows, patient_uid, folder, eye, "unparsed_eye", prefix)
                if bool(visit["date_is_future"]):
                    add_issue(rows, patient_uid, folder, eye, "future_exam_date", prefix)
                if int(visit["num_oct_raw_images"]) == 0:
                    add_issue(rows, patient_uid, folder, eye, "missing_visit_raw_image", prefix)
                if int(visit["num_2d_analysis_images"]) == 0:
                    add_issue(rows, patient_uid, folder, eye, "missing_visit_2d_analysis", prefix)

    for record in records:
        if record.modality in {"oct_raw", "oct_2d_analysis"} and (not record.can_parse_exam_date or not record.can_parse_eye):
            add_issue(
                rows,
                record.patient_uid,
                record.source_patient_folder,
                record.eye,
                "unparsed_oct_filename",
                safe_relative(record.path),
            )

    columns = ["patient_uid", "source_patient_folder", "eye", "issue_type", "detail"]
    return pd.DataFrame(rows, columns=columns).sort_values(["patient_uid", "eye", "issue_type"], kind="stable")


def write_summary(patient_df: pd.DataFrame, visit_df: pd.DataFrame, missing_df: pd.DataFrame, path: Path) -> None:
    strict_patients = int(patient_df["is_strict_patient_folder_name"].sum()) if not patient_df.empty else 0
    folder_count = len(patient_df)
    recognized_patients = int(patient_df["patient_uid"].nunique()) if not patient_df.empty else 0
    patient_ids = patient_df["patient_uid"].astype(str).tolist() if not patient_df.empty else []
    patient_range = f"{patient_ids[0]} - {patient_ids[-1]}" if patient_ids else ""
    eye_visit_df = visit_df[visit_df["eye"].isin(["OD", "OS"])] if not visit_df.empty else visit_df
    eye_keys = eye_visit_df[["patient_uid", "eye"]].drop_duplicates() if not eye_visit_df.empty else pd.DataFrame()
    preop_eye_keys = eye_visit_df[
        (eye_visit_df["is_preop"].astype(bool)) & (eye_visit_df["num_oct_raw_images"] > 0)
    ][["patient_uid", "eye"]].drop_duplicates() if not eye_visit_df.empty else pd.DataFrame()
    pod1_eye_keys = eye_visit_df[
        (eye_visit_df["is_postop"].astype(bool))
        & (eye_visit_df["postop_day"].astype(str) == "1")
        & (eye_visit_df["num_2d_analysis_images"] > 0)
    ][["patient_uid", "eye"]].drop_duplicates() if not eye_visit_df.empty else pd.DataFrame()
    date_parse_fail = int((~visit_df["can_parse_exam_date"].astype(bool)).sum()) if not visit_df.empty else 0
    eye_parse_fail = int((~visit_df["can_parse_eye"].astype(bool)).sum()) if not visit_df.empty else 0
    unknown_eye_records = int((visit_df["eye"].astype(str) == "unknown").sum()) if not visit_df.empty else 0
    nonstandard = patient_df.loc[~patient_df["is_strict_patient_folder_name"].astype(bool), "source_patient_folder"].astype(str).tolist() if not patient_df.empty else []
    issue_counts = missing_df["issue_type"].value_counts().to_dict() if not missing_df.empty else {}
    manual_issue_types = {
        "naming_anomaly",
        "duplicate_patient_uid",
        "duplicate_global_patient_uid",
        "missing_eye",
        "missing_preop_as_oct",
        "missing_pod1_2d_analysis",
        "future_exam_date",
        "unparsed_oct_filename",
    }
    manual_df = missing_df[missing_df["issue_type"].isin(manual_issue_types)] if not missing_df.empty else missing_df
    manual_targets = (
        manual_df[["patient_uid", "eye"]]
        .drop_duplicates()
        .assign(target=lambda df: df["patient_uid"].astype(str) + df["eye"].astype(str).map(lambda value: f"/{value}" if value else ""))
    )
    manual_list = sorted(manual_targets["target"].tolist()) if not manual_targets.empty else []
    can_proceed = recognized_patients > 0 and len(preop_eye_keys) > 0 and len(pod1_eye_keys) > 0

    lines = [
        "# Batch 04 intake QC summary",
        "",
        "本次 QC 只检查 `data/raw/real_export_batch_04/` 接入状态；未训练模型，未合并 combined manifest，未生成 patient-level split，未修改 batch_01 / batch_02 / batch_03 结果，也未修改论文正文。",
        "",
        "## 目录与 patient 识别",
        "",
        f"- patient 文件夹数量：{folder_count}",
        f"- 识别到的 patient_uid 数量：{recognized_patients}",
        f"- patient_uid 范围：{patient_range}",
        f"- 严格符合 `patient_数字` 命名的文件夹数：{strict_patients}",
        f"- 非标准 patient 文件夹名：{', '.join(nonstandard) if nonstandard else '无'}",
        "",
        "## 眼别与 visit 覆盖",
        "",
        f"- 实际识别眼数：{len(eye_keys)}",
        f"- 具备术前 AS-OCT raw 的眼数：{len(preop_eye_keys)}",
        f"- 具备 POD1 2DAnalysis label 候选的眼数：{len(pod1_eye_keys)}",
        "",
        "## 解析与重复风险",
        "",
        f"- 日期解析失败数量：{date_parse_fail}",
        f"- 眼别解析失败数量：{eye_parse_fail}",
        f"- unknown eye records 数量：{unknown_eye_records}",
        f"- 与 batch_01 / batch_02 / batch_03 重复的 patient_uid 数：{int(patient_df['duplicates_prior_patient_uid'].sum()) if not patient_df.empty else 0}",
        f"- 与既有 combined 重复的 global_patient_uid 数：{int(patient_df['duplicates_prior_global_patient_uid'].sum()) if not patient_df.empty else 0}",
        "",
        "## 主要 QC 发现",
        "",
        f"- issue_type 分布：{issue_counts if issue_counts else '未发现需记录的问题'}",
        f"- 需要人工检查的 patient / eye：{', '.join(manual_list) if manual_list else '无'}",
        "",
        "## 下一步判断",
        "",
        (
            "- batch_04 可以进入 initial manifest 构建和 2DAnalysis extraction。"
            if can_proceed
            else "- batch_04 暂不建议进入 initial manifest 构建或 extraction；请先处理上述缺失或解析问题。"
        ),
        "",
        "## 输出文件",
        "",
        "- `artifacts/reports/real_export_batch_04/intake_qc/batch04_patient_inventory.csv`",
        "- `artifacts/reports/real_export_batch_04/intake_qc/batch04_eye_visit_inventory.csv`",
        "- `artifacts/reports/real_export_batch_04/intake_qc/batch04_missing_data_report.csv`",
        "- `artifacts/reports/real_export_batch_04/intake_qc/batch04_intake_qc_summary.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Raw export root does not exist: {RAW_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patient_root = resolve_patient_root(RAW_ROOT)
    patient_dirs = sorted(path for path in patient_root.iterdir() if path.is_dir())
    prior_patient_uids, prior_global_patient_uids = load_prior_ids()

    records = collect_records(patient_dirs)
    visit_df = build_visit_inventory(records)
    patient_df = build_patient_inventory(
        patient_dirs=patient_dirs,
        records=records,
        visit_df=visit_df,
        prior_patient_uids=prior_patient_uids,
        prior_global_patient_uids=prior_global_patient_uids,
    )
    missing_df = build_missing_report(patient_df=patient_df, visit_df=visit_df, records=records)

    patient_df.to_csv(OUTPUT_DIR / "batch04_patient_inventory.csv", index=False, encoding="utf-8")
    visit_df.to_csv(OUTPUT_DIR / "batch04_eye_visit_inventory.csv", index=False, encoding="utf-8")
    missing_df.to_csv(OUTPUT_DIR / "batch04_missing_data_report.csv", index=False, encoding="utf-8")
    write_summary(patient_df, visit_df, missing_df, OUTPUT_DIR / "batch04_intake_qc_summary.md")

    eye_count = len(visit_df.loc[visit_df["eye"].isin(["OD", "OS"]), ["patient_uid", "eye"]].drop_duplicates()) if not visit_df.empty else 0
    print(f"Patient folders: {len(patient_dirs)}")
    print(f"Recognized patient_uids: {patient_df['patient_uid'].nunique()}")
    print(f"Eyes detected: {eye_count}")
    print(f"QC outputs: {safe_relative(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
