
from dotenv import load_dotenv
import os

from navec import Navec
import pymorphy3

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
IS_PROD = os.getenv('PROD') == 'prod'
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ACCESS_TOKEN_EXPIRE_DAYS = 7
CERTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "certs")

MORPHY = pymorphy3.MorphAnalyzer(lang='ru')
_ROOT_DIR = os.path.split( os.path.dirname(__file__))[0]
PROJECT_ROOT_DIR = os.environ.get("PROJECT_ROOT_DIR", None) or os.path.dirname(_ROOT_DIR)
_navec_path = os.path.join(PROJECT_ROOT_DIR, 'neural', 'navec', 'navec_hudlit_v1_12B_500K_300d_100q.tar')
NAVEC = Navec.load(_navec_path)

if IS_PROD:
    from deeppavlov.core.commands.infer import build_model
    DEEP_PAVLOV_MODEL = build_model(os.path.join(PROJECT_ROOT_DIR, "neural", "deeppavlov", "configs", "squad", "ru_odqa_infer_wiki.json"), install=False, download=False)
else:
    DEEP_PAVLOV_MODEL = None
