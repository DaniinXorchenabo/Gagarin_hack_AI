from os.path import join, dirname
import traceback
import nltk
import ssl

from deeppavlov import configs, build_model
from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

proj_root_path = '/workspace/project'
try:
    model = build_model(join(proj_root_path, "neural", "deeppavlov", "configs", "squad", "ru_odqa_infer_wiki.json"),
                        install=True,
                        download=False)
except Exception as e:
    print(e)
    print(traceback.format_exc())



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# nltk.download()
nltk.download('punkt')

nltk.download('stopwords')
nltk.download('corpus')

