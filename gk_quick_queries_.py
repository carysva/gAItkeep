"""
gk_quick_queries.py

Quick launcher for querying multiple test images with run_gk_two_brand.bat.

Examples:
  python gk_quick_queries.py
  python gk_quick_queries.py --run
  python gk_quick_queries.py --run --threshold 0.9
  python gk_quick_queries.py --run --only 1,3,7
  python gk_quick_queries.py --run --batch m
  python gk_quick_queries.py --run --only 31,32,33,34,35,36,37,38,39
  python gk_quick_queries.py --run --batch l
  python gk_quick_queries.py --run --only 40,41,42,43,44,45,46,47,48,49,50
  python gk_quick_queries.py --run --batch z
  python gk_quick_queries.py --run --only 51,52,53,54,55,56,57,58,59,60
  python gk_quick_queries.py --run --batch d
  python gk_quick_queries.py --run --only 61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76
  python gk_quick_queries.py --run --batch w
  python gk_quick_queries.py --run --only 77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102
  python gk_quick_queries.py --run --batch h
  python gk_quick_queries.py --run --only 103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

import gen_ai_project.gk_two_brand_matcher as gk


try:
    ROOT_DIR = Path(__file__).resolve().parent
except NameError:
    # Supports IPython/interactive execution where __file__ is undefined.
    ROOT_DIR = Path.cwd()

# Test images: prefer repo `assets/`, else Cursor workspace storage (same filenames).
CURSOR_ASSETS_FALLBACK = Path(
    r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets"
)


def resolve_test_image(filename: str) -> str:
    """Return a path string to a test image; tries BZAN583/assets then Cursor assets."""
    for base in (ROOT_DIR / "assets", CURSOR_ASSETS_FALLBACK):
        p = base / filename
        if p.exists():
            return str(p)
    return str(ROOT_DIR / "assets" / filename)


DEFAULT_LOG_CSV = ROOT_DIR / "gk_two_brand_meta" / "quick_queries_log.csv"
TUNING_JOURNAL_CSV = ROOT_DIR / "gk_two_brand_meta" / "tuning_journal.csv"
CSV_COLUMNS = [
    "run_at_utc",
    "run_group",
    "case_id",
    "case_order",
    "case_label",
    "photo_name",
    "image_filename",
    "image_path",
    "image_path_resolved",
    "ok",
    "predicted_brand",
    "same_photo",
    "item_match",
    "decision_label",
    "reason",
    "similarity",
    "threshold",
    "item_match_threshold",
    "catalog_warning",
    "catalog_brand_counts",
    "best_match_brand",
    "best_match_path",
    "best_match_sha256",
    "user_message",
    "debug_error_ref",
    "error",
    "user_brand_correct_checkbox",
    "user_brand_correct",
    "error_tag_notes",
]

# Optional aliases for newest batch when files were renamed.
RENAMED_ALIAS_BY_CASE: dict[str, str] = {
    "case_21": "k1",
    "case_22": "k2",
    "case_23": "k3",
    "case_24": "k4",
    "case_25": "k5",
    "case_26": "kk1",
    "case_27": "kk2",
    "case_28": "kk3",
    "case_29": "kk4",
    "case_30": "kk5",
    # Batch m / mm (Frankies vs Mirchi manual grading by filename prefix)
    "case_31": "m1",
    "case_32": "m2",
    "case_33": "m3",
    "case_34": "m4",
    "case_35": "mm1",
    "case_36": "mm2",
    "case_37": "mm3",
    "case_38": "mm4",
    "case_39": "mm5",
    # Batch L / LL (lifestyle / UGC-style test photos; grade by filename prefix)
    "case_40": "L1",
    "case_41": "L2",
    "case_42": "L3",
    "case_43": "L4",
    "case_44": "L5",
    "case_45": "L6",
    "case_46": "LL1",
    "case_47": "LL2",
    "case_48": "LL3",
    "case_49": "LL4",
    "case_50": "LL5",
    "case_51": "z1",
    "case_52": "z2",
    "case_53": "z3",
    "case_54": "z4",
    "case_55": "z5",
    "case_56": "zz1",
    "case_57": "zz2",
    "case_58": "zz3",
    "case_59": "zz4",
    "case_60": "zz5",
    "case_61": "d1",
    "case_62": "d2",
    "case_63": "d3",
    "case_64": "d4",
    "case_65": "d5",
    "case_66": "d6",
    "case_67": "dd1",
    "case_68": "dd2",
    "case_69": "dd3",
    "case_70": "dd4",
    "case_71": "dd5",
    "case_72": "dd6",
    "case_73": "dd7",
    "case_74": "dd8",
    "case_75": "dd9",
    "case_76": "dd10",
    "case_77": "w1",
    "case_78": "w2",
    "case_79": "w3",
    "case_80": "w4",
    "case_81": "w5",
    "case_82": "w6",
    "case_83": "w7",
    "case_84": "w8",
    "case_85": "w9",
    "case_86": "w10",
    "case_87": "w11",
    "case_88": "w12",
    "case_89": "w13",
    "case_90": "w1_",
    "case_91": "w2_",
    "case_92": "w3_",
    "case_93": "w4_",
    "case_94": "w5_",
    "case_95": "w6_",
    "case_96": "w7_",
    "case_97": "w8_",
    "case_98": "w9_",
    "case_99": "w10_",
    "case_100": "w11_",
    "case_101": "w12_",
    "case_102": "w13_",
    "case_103": "h1",
    "case_104": "h2",
    "case_105": "h3",
    "case_106": "h4",
    "case_107": "h5",
    "case_108": "h6",
    "case_109": "h7",
    "case_110": "h8",
    "case_111": "h1_",
    "case_112": "h2_",
    "case_113": "h3_",
    "case_114": "h4_",
    "case_115": "h5_",
    "case_116": "h6_",
    "case_117": "h7_",
    "case_118": "h8_",
}

# Backfill tags for known Mirchi→Frankies nearest-neighbor confusions (L-batch).
PRESET_ERROR_TAGS_BY_PHOTO: dict[str, str] = {
    "LL1": "mbk_as_fb_nn",
    "LL4": "mbk_as_fb_nn",
    "LL5": "mbk_as_fb_nn",
    "zz2": "mbk_as_fb_nn",
    "zz3": "mbk_as_fb_nn",
    "zz4": "mbk_as_fb_nn",
    "zz5": "mbk_as_fb_nn",
}

_M_BATCH_FILENAMES = [
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_m1-38bdeac4-214f-468c-af97-d5e6351dba39.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_m2-8bc5cbeb-8ac6-4160-a0f1-0083bc926469.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_m3-aeed0aca-8f01-4def-bdd2-aa5c1a104297.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_m4-a3855463-d998-4460-819d-4a23157cf214.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_mm1-b4ac56dd-4be5-4d15-b55a-90907bdb5614.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_mm2-66472324-c2cf-4fae-9f82-334f291db5b3.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_mm3-af2ec63e-c5a9-4591-973a-3f13f960570c.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_mm4-9a90c630-540b-42d1-9274-61f54ba5ce71.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_mm5-3d40f8ba-560e-48ce-9a24-6f970b8b7b6c.png",
]

_L_LL_BATCH_FILENAMES = [
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L1-4a0d68ce-eb6a-432a-84d4-ebf61f502594.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L2-f198616f-a200-4b18-85df-fb59baee803d.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L3-cd5b8a24-b20d-422c-a0e2-976a94b14355.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L4-648c37dd-6cdd-43fe-8f88-02265f5dd477.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L5-4e26544e-392d-4df7-af66-01edfe4723ba.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_L6-93f7d56b-0cbf-49fc-81d2-7a78fc5a96ac.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_LL1-9dcdadad-12e6-4303-bbe4-9eaed39fce62.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_LL2-6825279a-826f-4076-ad81-4af02bfcf0b1.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_LL3-f1c1ab50-1e63-4494-8a1f-c308c66ecbdf.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_LL4-8df6db6c-197e-4df0-80ea-575d16f73f41.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_LL5-0cbf0d4c-2005-47c0-9af5-d107fc544cb8.png",
]

_Z_ZZ_BATCH_FILENAMES = [
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_z1-75bad0da-a40d-4570-9972-63e945c17302.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_z2-be008c79-d39d-4162-9489-bcc24a3393b7.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_z3-51ff8a17-d373-4b7f-8e13-702723c32f7a.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_z4-74236a87-43bb-4ad4-9561-7e83d433cdd3.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_z5-1da67482-bdd6-4bb5-9ed4-5bdaec90ed91.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_zz1-f8b39ef9-be7c-4ae1-b37e-71e0514523dd.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_zz2-ccd04499-6c38-4974-9a73-f7ac3143caf8.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_zz3-0be00961-ac0b-46a6-aef1-718b4f4995e1.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_zz4-8c00c13a-25c9-4ea0-8e63-95c600ab2c81.png",
    "c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_zz5-7a872822-b062-4ddc-89a3-499de15c2325.png",
]

_D_DD_BATCH_FILENAMES = [
    "d1.png",
    "d2.png",
    "d3.png",
    "d4.png",
    "d5.png",
    "d6.png",
    "dd1.png",
    "dd2.png",
    "dd3.png",
    "dd4.png",
    "dd5.png",
    "dd6.png",
    "dd7.png",
    "dd8.png",
    "dd9.png",
    "dd10.png",
]

_W_BATCH_FILENAMES = [
    "w1.png",
    "w2.png",
    "w3.png",
    "w4.png",
    "w5.png",
    "w6.png",
    "w7.png",
    "w8.png",
    "w9.png",
    "w10.png",
    "w11.png",
    "w12.png",
    "w13.png",
    "w1_.png",
    "w2_.png",
    "w3_.png",
    "w4_.png",
    "w5_.png",
    "w6_.png",
    "w7_.png",
    "w8_.png",
    "w9_.png",
    "w10_.png",
    "w11_.png",
    "w12_.png",
    "w13_.png",
]

_H_BATCH_FILENAMES = [
    "h1.png",
    "h2.png",
    "h3.png",
    "h4.png",
    "h5.png",
    "h6.png",
    "h7.png",
    "h8.png",
    "h1_.png",
    "h2_.png",
    "h3_.png",
    "h4_.png",
    "h5_.png",
    "h6_.png",
    "h7_.png",
    "h8_.png",
]

# Images shared in chat and stored in workspace assets.
IMAGE_CASES: list[tuple[str, str]] = [
    ("case_01", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-f3159b27-335b-4ac8-bb96-3587dbc27ca0.png"),
    ("case_02", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-7390ed98-0ec7-4d88-acc0-0c3f8777356f.png"),
    ("case_03", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-3a012193-e42f-4bdf-b9bd-9bc991b7338b.png"),
    ("case_04", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-3be685f8-0404-4e33-bf01-670019dad6cb.png"),
    ("case_05", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-83d2b916-348f-499e-b840-35fdfdb39769.png"),
    ("case_06", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-1b3489fd-14e5-40e0-9e73-9c7884dfa7fe.png"),
    ("case_07", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-f804ff95-4665-4197-b810-9e7da401e212.png"),
    ("case_08", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-df456051-19d1-40ca-8ada-9ebdc3250432.png"),
    ("case_09", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-5d87a0f2-8705-424b-b7b6-de826cb4ebaf.png"),
    ("case_10", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-50b0de75-ec12-4371-8f87-8f408141f001.png"),
    ("case_11", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-7d1d29ff-bbc8-4944-b122-98b4d663f1c4.png"),
    ("case_12", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-5166aae5-accf-453d-b528-c1905b35f26b.png"),
    ("case_13", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-42dc749c-3263-4881-acef-b393f6f20e08.png"),
    ("case_14", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-2d2b00a1-bd55-4f36-b669-7f1e2dbdedbd.png"),
    ("case_15", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-f28c7636-2f9c-4cc1-9da8-ebc38ab88be7.png"),
    ("case_16", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-67271afd-5d59-47e3-afdb-9cb303ba539a.png"),
    ("case_17", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-593e451c-41b1-4288-b176-c01358337afe.png"),
    ("case_18", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-a831e193-55de-4c82-bf3a-3b7ea7e7010e.png"),
    ("case_19", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-b4d486e4-f019-482f-9da0-4fc356c6ce42.png"),
    ("case_20", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_image-c122f68b-b3f0-426d-baf4-782974fb5e97.png"),
    ("case_21", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_k1-7e9e8f2b-fcc3-4e24-a4fd-194b5478a81b.png"),
    ("case_22", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_k2-ac7e4eff-b29e-43fb-91ca-08e448103438.png"),
    ("case_23", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_k3-d0a7b0b1-4e96-4bf6-aed3-2832cadd926e.png"),
    ("case_24", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_k4-51c1f7e2-f2bb-4b4e-add7-d945abdb351f.png"),
    ("case_25", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_k5-9b820f05-7bc2-4b82-abfd-7d7f966c1cf6.png"),
    ("case_26", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_kk1-92c704ec-8678-4908-8249-eff3157e7338.png"),
    ("case_27", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_kk2-89ceb1c2-8ca8-42c1-a41c-0d90d8779652.png"),
    ("case_28", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_kk3-c981001a-3ef6-47e2-b0ba-78254cede7ad.png"),
    ("case_29", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_kk4-d46945d7-c719-4a11-8548-038cf3b22641.png"),
    ("case_30", r"C:\Users\carys\.cursor\projects\c-Users-carys-OneDrive-Desktop-BZAN554-BZAN583\assets\c__Users_carys_AppData_Roaming_Cursor_User_workspaceStorage_329fd904182f3ea780cca0e692525314_images_kk5-348c9d61-f8f7-477c-ad8d-c0937d46ca86.png"),
    ("case_31", resolve_test_image(_M_BATCH_FILENAMES[0])),
    ("case_32", resolve_test_image(_M_BATCH_FILENAMES[1])),
    ("case_33", resolve_test_image(_M_BATCH_FILENAMES[2])),
    ("case_34", resolve_test_image(_M_BATCH_FILENAMES[3])),
    ("case_35", resolve_test_image(_M_BATCH_FILENAMES[4])),
    ("case_36", resolve_test_image(_M_BATCH_FILENAMES[5])),
    ("case_37", resolve_test_image(_M_BATCH_FILENAMES[6])),
    ("case_38", resolve_test_image(_M_BATCH_FILENAMES[7])),
    ("case_39", resolve_test_image(_M_BATCH_FILENAMES[8])),
    *[("case_%d" % (40 + j), resolve_test_image(fn)) for j, fn in enumerate(_L_LL_BATCH_FILENAMES)],
    *[("case_%d" % (51 + j), resolve_test_image(fn)) for j, fn in enumerate(_Z_ZZ_BATCH_FILENAMES)],
    *[("case_%d" % (61 + j), resolve_test_image(fn)) for j, fn in enumerate(_D_DD_BATCH_FILENAMES)],
    *[("case_%d" % (77 + j), resolve_test_image(fn)) for j, fn in enumerate(_W_BATCH_FILENAMES)],
    *[("case_%d" % (103 + j), resolve_test_image(fn)) for j, fn in enumerate(_H_BATCH_FILENAMES)],
]

# Case index numbers for the m/mm batch only (use with --batch m).
M_MM_BATCH_CASE_NUMBERS = list(range(31, 40))

# Case index numbers for the L/LL lifestyle batch (use with --batch l).
L_LL_BATCH_CASE_NUMBERS = list(range(40, 51))

# Case index numbers for the z/zz batch (use with --batch z).
Z_ZZ_BATCH_CASE_NUMBERS = list(range(51, 61))

# Case index numbers for the d/dd batch (use with --batch d).
D_DD_BATCH_CASE_NUMBERS = list(range(61, 77))

# Case index numbers for the w/w_ batch (use with --batch w).
W_BATCH_CASE_NUMBERS = list(range(77, 103))

# Case index numbers for the h/h_ batch (use with --batch h).
H_BATCH_CASE_NUMBERS = list(range(103, 119))


def parse_only(raw: str) -> set[int]:
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def build_cmd(image_path: str, threshold: float) -> list[str]:
    return ["query_same_photo", image_path, str(threshold)]


def _normalize_row(row: dict[str, str], fieldnames: list[str]) -> dict[str, str]:
    return {k: str(row.get(k, "")) for k in fieldnames}


def resolve_image_path(label: str, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    alias = RENAMED_ALIAS_BY_CASE.get(label)
    if not alias:
        return p
    search_roots = [
        ROOT_DIR,
        Path(r"C:\Users\carys\OneDrive\Desktop"),
        Path(r"C:\Users\carys\OneDrive\Pictures"),
        Path(r"C:\Users\carys\.cursor\projects"),
    ]
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for ext in exts:
            # Support Cursor asset naming like "...images_k1-<uuid>.png" or "...images_m1-..."
            hits.extend(root.rglob(f"*{alias}*{ext}"))
    if hits:
        # pick most recently modified if multiple
        hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return hits[0]
    return p


def display_photo_name(label: str, resolved_path: Path, raw_path: str) -> str:
    alias = RENAMED_ALIAS_BY_CASE.get(label)
    if alias:
        return alias
    name = (resolved_path.name if resolved_path.exists() else Path(raw_path).name).lower()
    # LL before L so "LL1" is not parsed as L + L1
    m = re.search(r"\b(h\d+_?|w\d+_?|zz\d+|z\d+|LL\d+|L\d+|kk\d+|k\d+|mm\d+|m\d+)\b", name)
    if m:
        return m.group(1)
    # Cursor pattern: images_h1_-..., images_h2-..., images_zz1-..., etc.
    m2h_ = re.search(r"images_(h\d+_)-", name)
    if m2h_:
        return m2h_.group(1)
    m2h = re.search(r"images_(h\d+)-", name)
    if m2h:
        return m2h.group(1)
    m2 = re.search(r"images_(w\d+_?|zz\d+|z\d+|LL\d+|L\d+|kk\d+|k\d+|mm\d+|m\d+)", name)
    if m2:
        return m2.group(1)
    return (resolved_path.stem if resolved_path.exists() else Path(raw_path).stem)


def migrate_quick_queries_log(out_csv: Path) -> None:
    """Ensure CSV matches CSV_COLUMNS; backfill error_tag_notes for known LL false Frankies."""
    if not out_csv.exists():
        return
    fieldnames = CSV_COLUMNS
    with out_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        existing = list(reader)
    norm = [_normalize_row(r, fieldnames) for r in existing]
    changed = False
    for r in norm:
        if r.get("run_group") == "HEADER_ROW":
            if "error_tag_notes" not in old_fields:
                for k in fieldnames:
                    r[k] = k
                changed = True
            continue
        if not r.get("case_id", "").strip().isdigit():
            continue
        if r.get("error_tag_notes", "").strip():
            continue
        ph = r.get("photo_name", "")
        if ph in PRESET_ERROR_TAGS_BY_PHOTO and r.get("predicted_brand") == "FrankiesBikinis":
            r["error_tag_notes"] = PRESET_ERROR_TAGS_BY_PHOTO[ph]
            changed = True
    need_rewrite = changed or ("error_tag_notes" not in old_fields)
    if not need_rewrite:
        return
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(norm)


def export_quick_queries_log_xlsx(csv_path: Path) -> Path | None:
    """Write a sibling .xlsx mirror of the CSV (requires openpyxl)."""
    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        from openpyxl import Workbook
    except ImportError:
        print("Note: install openpyxl to auto-write quick_queries_log.xlsx (pip install openpyxl)")
        return None
    wb = Workbook()
    ws = wb.active
    ws.title = "quick_queries_log"
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(xlsx_path)
    return xlsx_path


def append_rows(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = CSV_COLUMNS

    existing_rows: list[dict[str, str]] = []
    if out_csv.exists():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    normalized_existing = [_normalize_row(r, fieldnames) for r in existing_rows]
    normalized_new = [_normalize_row(r, fieldnames) for r in rows]

    # Add visual separation between batch runs in Excel-friendly CSV.
    sep_run = rows[0].get("run_at_utc", "")
    separator_rows = [
        _normalize_row({"run_at_utc": sep_run, "run_group": "----- NEW RUN -----"}, fieldnames),
        _normalize_row({"run_group": "HEADER_ROW", **{k: k for k in fieldnames}}, fieldnames),
        _normalize_row({}, fieldnames),
    ]

    all_rows = normalized_existing + separator_rows + normalized_new
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)


def append_tuning_journal(run_stamp: str, out_csv: Path, cases_run: int) -> None:
    cols = [
        "ts_utc",
        "same_photo_threshold",
        "item_match_threshold",
        "brand_vote_top_k",
        "brand_vote_per_brand_top_m",
        "notes",
        "observed_result",
        "summary_csv",
    ]
    row = {
        "ts_utc": run_stamp,
        "same_photo_threshold": str(gk.SAME_PHOTO_THRESHOLD),
        "item_match_threshold": str(gk.ITEM_MATCH_THRESHOLD),
        "brand_vote_top_k": str(gk.BRAND_VOTE_TOP_K),
        "brand_vote_per_brand_top_m": str(gk.BRAND_VOTE_PER_BRAND_TOP_M),
        "notes": "auto-log from gk_quick_queries.py",
        "observed_result": f"completed {cases_run} cases",
        "summary_csv": str(out_csv),
    }
    exists = TUNING_JOURNAL_CSV.exists()
    with TUNING_JOURNAL_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow(row)


def preflight_missing_cases(only_ids: set[int] | None) -> list[tuple[int, str, str]]:
    missing: list[tuple[int, str, str]] = []
    for i, (label, img_path) in enumerate(IMAGE_CASES, start=1):
        if only_ids and i not in only_ids:
            continue
        resolved = resolve_image_path(label, img_path)
        if not resolved.exists():
            missing.append((i, label, img_path))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick query helper for GK image tests.")
    parser.add_argument("--run", action="store_true", help="Execute commands instead of only printing.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=gk.SAME_PHOTO_THRESHOLD,
        help="same_photo threshold passed to run_gk_two_brand.bat",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated case numbers to run/print (e.g., 1,3,10).",
    )
    parser.add_argument(
        "--batch",
        choices=["", "m", "l", "z", "d", "w", "h"],
        default="",
        help="Shorthand: m=31-39 (m/mm), l=40-50 (L/LL), z=51-60 (z/zz), d=61-76 (d/dd), w=77-102 (w/w_), h=103-118 (h/h_).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_LOG_CSV),
        help="CSV file path for logging results when --run is used.",
    )
    args = parser.parse_args()

    only_ids = parse_only(args.only) if args.only else None
    if args.batch in ("m", "l", "z", "d", "w", "h"):
        if args.only:
            parser.error("Use either --batch or --only=..., not both.")
        if args.batch == "m":
            only_ids = set(M_MM_BATCH_CASE_NUMBERS)
        elif args.batch == "l":
            only_ids = set(L_LL_BATCH_CASE_NUMBERS)
        elif args.batch == "z":
            only_ids = set(Z_ZZ_BATCH_CASE_NUMBERS)
        elif args.batch == "d":
            only_ids = set(D_DD_BATCH_CASE_NUMBERS)
        elif args.batch == "w":
            only_ids = set(W_BATCH_CASE_NUMBERS)
        else:
            only_ids = set(H_BATCH_CASE_NUMBERS)
    missing = preflight_missing_cases(only_ids)
    if missing:
        print("\nMissing image paths detected (update IMAGE_CASES paths):")
        for i, label, p in missing:
            print(f"  - [{i:02d}] {label}: {p}")
        print("")
    rows: list[dict[str, str]] = []
    total = 0
    run_stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    run_group = run_stamp
    out_csv = Path(args.out)
    if args.run:
        migrate_quick_queries_log(out_csv)

    for i, (label, img_path) in enumerate(IMAGE_CASES, start=1):
        if only_ids and i not in only_ids:
            continue
        total += 1
        resolved_path = resolve_image_path(label, img_path)
        cmd = build_cmd(img_path, args.threshold)
        cmd_preview = f'{cmd[0]} --image "{resolved_path}" --threshold {args.threshold}'
        print(f"\n[{i:02d}] {label}")
        print(cmd_preview)
        if args.run:
            print("Running...")
            out = gk.query_same_photo(resolved_path, threshold=args.threshold)
            print(json.dumps(out, indent=2))
            best = out.get("best_match") or {}
            rows.append(
                {
                    "run_at_utc": run_stamp,
                    "run_group": run_group,
                    "case_id": str(i),
                    "case_order": str(i),
                    "case_label": label,
                    "photo_name": display_photo_name(label, resolved_path, img_path),
                    "image_filename": resolved_path.name,
                    "image_path": img_path,
                    "image_path_resolved": str(resolved_path),
                    "ok": str(out.get("ok")),
                    "predicted_brand": str(out.get("predicted_brand")),
                    "same_photo": str(out.get("same_photo")),
                    "item_match": str(out.get("item_match")),
                    "decision_label": str(out.get("decision_label")),
                    "reason": str(out.get("reason")),
                    "similarity": str(out.get("similarity")),
                    "threshold": str(out.get("threshold")),
                    "item_match_threshold": str(out.get("item_match_threshold")),
                    "catalog_warning": str(out.get("catalog_warning")),
                    "catalog_brand_counts": str(out.get("catalog_brand_counts")),
                    "best_match_brand": str(best.get("brand")),
                    "best_match_path": str(best.get("path")),
                    "best_match_sha256": str(best.get("sha256")),
                    "user_message": str(out.get("user_message", "")),
                    "debug_error_ref": str(out.get("debug_error_ref", "")),
                    "error": str(out.get("error", "")),
                    "user_brand_correct_checkbox": "[ ]",
                    "user_brand_correct": "",
                    "error_tag_notes": "",
                }
            )

    if args.run and rows:
        append_rows(rows, out_csv)
        append_tuning_journal(run_stamp, out_csv, len(rows))
        xlsx_path = export_quick_queries_log_xlsx(out_csv)
        print(f"\nCompleted {len(rows)}/{total} cases.")
        print(f"Appended {len(rows)} rows to: {out_csv}")
        if xlsx_path:
            print(f"Excel mirror: {xlsx_path}")
        print(f"Auto-updated tuning journal: {TUNING_JOURNAL_CSV}")


if __name__ == "__main__":
    main()

