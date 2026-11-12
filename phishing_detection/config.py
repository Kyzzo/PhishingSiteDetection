"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

DATASET_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
RAW_CSV = RAW_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"

# Original label: 1 = legitimate, 0 = phishing.
# We flip it so phishing is the positive class (1) for precision/recall/F1.
RAW_TARGET = "label"
TARGET = "is_phishing"

# Identifier / raw-text columns excluded from modeling.
# Provisional: the proposal says we examine the text/categorical columns and encode or
# exclude them per model. Joshua confirms or changes this list in Week 1 (see SCHEDULE.md).
DROP_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title"]

# Feature sets. "full" keeps every usable feature (comparable to the PhiUSIIL paper).
# "ablated" keeps only features computed from the URL string, minus the two that give the
# label away: URLSimilarityIndex alone separates the classes at 99.7% accuracy, and every
# legitimate URL is HTTPS. On the full set every model scores ~100% F1, so the comparison
# says nothing; on the ablated set LR / DT / RF reach 0.973 / 0.990 / 0.994 CV F1.
# Caveat: the remaining URL features still partly encode the "https://www." prefix that
# all legitimate URLs share (see README, "Feature sets").
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
    "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]
ABLATED_EXCLUDE = ["URLSimilarityIndex", "IsHTTPS"]
FEATURE_SETS = {
    "full": None,  # None = keep every column clean() returns
    "ablated": [c for c in URL_FEATURES if c not in ABLATED_EXCLUDE],
}
FEATURE_SET = "full"  # default used by preprocess.train_test / load_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = -1

# Primary metric for model selection and tuning.
SCORING = "f1"
