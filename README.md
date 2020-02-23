# toolbox
Curated libraries for a faster workflow

# Phase: Data
## Data Annotation
- Image: [makesense.ai](https://www.makesense.ai/) 
- Text: [doccano](https://doccano.herokuapp.com/), [prodigy](https://prodi.gy/), [dataturks](https://dataturks.com/), [brat](http://brat.nlplab.org/)

## Datasets
- Text: [nlp-datasets](https://github.com/niderhoff/nlp-datasets), [curse-words](https://github.com/reimertz/curse-words), [badwords](https://github.com/MauriceButler/badwords), [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words), [english-words (A text file containing over 466k English words)](https://github.com/dwyl/english-words), [10K most common words](https://github.com/first20hours/google-10000-english), [1 trillion n-grams](https://catalog.ldc.upenn.edu/LDC2006T13), [The Big Bad NLP Database](https://quantumstat.com/dataset/dataset.html), [project gutenberg](https://www.gutenberg.org/), [oscar (big multilingual corpus)](https://traces1.inria.fr/oscar/)
- Image: [1 million fake faces](https://archive.org/details/1mFakeFaces)
- Dataset search engine: [datasetlist](https://www.datasetlist.com/), [UCI Machine Learning Datasets](https://archive.ics.uci.edu/ml/datasets.php), [Google Dataset Search](https://toolbox.google.com/datasetsearch), [fastai-datasets](https://course.fast.ai/datasets.html)

## Importing Data
- Audio: [pydub](https://github.com/jiaaro/pydub)
- Video: [pytube (download youtube vidoes)](https://github.com/nficano/pytube), [moviepy](https://zulko.github.io/moviepy/)
- Image: [py-image-dataset-generator (auto fetch images from web for certain search)](https://github.com/tomahim/py-image-dataset-generator)
- News: [news-please](https://github.com/fhamborg/news-please)
- PDF: [camelot](https://camelot-py.readthedocs.io/en/master/), [tabula-py](https://github.com/chezou/tabula-py), [Parsr](https://github.com/axa-group/Parsr), [pdftotext](https://pypi.org/project/pdftotext/)
- Excel: [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
- Remote file: [smart_open](https://github.com/RaRe-Technologies/smart_open)
- Crawling: [pyppeteer (chrome automation)](https://github.com/miyakogi/pyppeteer), [MechanicalSoup](https://github.com/MechanicalSoup/MechanicalSoup), [libextract](https://github.com/datalib/libextract)
- Google sheets: [gspread](https://github.com/burnash/gspread)
- Google drive: [gdown](https://github.com/wkentaro/gdown)
- Python API for datasets: [pydataset](https://github.com/iamaziz/PyDataset)
- Google maps location data: [geo-heatmap](https://github.com/luka1199/geo-heatmap)
- Tex to Speech: [gtts](https://github.com/pndurette/gTTS)
- Databases: [blaze (pandas and numpy interface to databases)](https://github.com/blaze/blaze)

## Data Augmentation
- Text: [nlpaug](https://github.com/makcedward/nlpaug), [noisemix](https://github.com/noisemix/noisemix)
- Image: [imgaug](https://github.com/aleju/imgaug/), [albumentations](https://github.com/albumentations-team/albumentations), [augmentor](https://github.com/mdbloice/Augmentor)
- Audio: [audiomentations](https://github.com/iver56/audiomentations), [muda](https://github.com/bmcfee/muda)
- OCR data: [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)

# Phase: Exploration

##  Data Preparation
- Missing values: [missingno](https://github.com/ResidentMario/missingno)
- Split images into train/validation/test: [split-folders](https://github.com/jfilter/split-folders)
- Class Imbalance: [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/api.html)
- Categorical encoding: [category_encoders](https://contrib.scikit-learn.org/categorical-encoding/index.html)
- Numerical data: [numerizer (convert natural language numerics into ints and floats)](https://github.com/jaidevd/numerizer)
- Data Validation: [pandera (validation for pandas)](https://github.com/pandera-dev/pandera)
- Data Cleaning: [pyjanitor (janitor ported to python)](https://github.com/ericmjl/pyjanitor)
- Parsing: [pyparsing](https://pyparsing-docs.readthedocs.io/en/latest/index.html)
- Natural date parser: [dateparser](https://github.com/scrapinghub/dateparser)
- Unicode: [text-unidecode](https://pypi.org/project/text-unidecode/)
- Emoji: [emoji](https://pypi.org/project/emoji/)
- Weak Supervision: [snorkel](https://www.snorkel.org/get-started/)

## Data Exploration
- View Jupyter notebooks through CLI: [nbdime](https://github.com/jupyter/nbdime)
- Parametrize notebooks: [papermill](https://github.com/nteract/papermill)
- Access notebooks programatically: [nbformat](https://nbformat.readthedocs.io/en/latest/api.html)
- Convert notebooks to other formats: [nbconvert](https://nbconvert.readthedocs.io/en/latest/)
- Extra utilities not present in frameworks: [mlxtend](https://github.com/rasbt/mlxtend)
- Maps in notebooks: [ipyleaflet](https://github.com/jupyter-widgets/ipyleaflet)
- Data Exploration: [bamboolib (a GUI for pandas)](https://bamboolib.8080labs.com/)

# Phase: Feature Engineering
## Feature Generation
- Automatic feature engineering: [featuretools](https://github.com/FeatureLabs/featuretools), [autopandas](https://autopandas.io/), [tsfresh (automatic feature engineering for time series)](https://github.com/blue-yonder/tsfresh)
- Custom distance metric learning: [metric-learn](http://contrib.scikit-learn.org/metric-learn/getting_started.html), [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- Time series: [python-holidays](https://github.com/dr-prodigy/python-holidays), [skits](https://github.com/ethanrosenthal/skits)
- DAG based dataset generation: [DFFML](https://intel.github.io/dffml/usage/integration.html)

# Phase: Modeling

## Model Selection
- Bruteforce through all scikit-learn model and parameters: [auto-sklearn](https://automl.github.io/auto-sklearn), [tpot](https://github.com/EpistasisLab/tpot)
- Autogenerate ML code: [automl-gs](https://github.com/minimaxir/automl-gs), [mindsdb](https://github.com/mindsdb/mindsdb), [autocat (auto-generate text classification models in spacy)](https://autocat.apps.allenai.org/)
- ML from command line (or Python or HTTP): [DFFML](https://intel.github.io/dffml/)
- Pretrained models: [modeldepot](https://modeldepot.io/browse), [pytorch-hub](https://pytorch.org/hub), [papers-with-code](https://paperswithcode.com/sota), [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- Find SOTA models: [sotawhat](https://sotawhat.herokuapp.com)
- Gradient Boosting: [catboost](https://catboost.ai/docs/concepts/about.html), [lightgbm (GPU-capable)](https://github.com/Microsoft/LightGBM), [thunderbm (GPU-capable)](https://github.com/Xtra-Computing/thundergbm)
- Hidden Markov Models: [hmmlearn](https://github.com/hmmlearn/hmmlearn)
- Genetic Programming: [gplearn](https://gplearn.readthedocs.io/en/stable/index.html)
- Active Learning: [modal](https://github.com/modAL-python/modAL)
- Support Vector Machines: [thundersvm (GPU-capable)](https://github.com/Xtra-Computing/thundersvm)
- Rule based classifier: [sklearn-expertsys](https://github.com/tmadl/sklearn-expertsys)
- Probabilistic modeling: [pomegranate](https://github.com/jmschrei/pomegranate)
- Graph Embedding and Community Detection: [karateclub](https://github.com/benedekrozemberczki/karateclub)
- Anomaly detection: [adtk](https://arundo-adtk.readthedocs-hosted.com/en/stable/install.html)
- Spiking Neural Network: [norse](https://github.com/norse/norse)
- Fuzzy Learning: [fylearn](https://github.com/sorend/fylearn), [scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy)
- Dimensionality reduction: [fbpca](https://github.com/facebook/fbpca)
- Noisy Label Learning: [cleanlab](https://github.com/cgnorthcutt/cleanlab)

## NLP
- Libraries: [spacy](https://spacy.io/) , [nltk](https://github.com/nltk/nltk), [corenlp](https://stanfordnlp.github.io/CoreNLP/), [deeppavlov](http://docs.deeppavlov.ai/en/master/index.html), [kashgari](https://kashgari.bmio.net/), [camphr (spacy plugin for transformers, elmo, udify)](https://github.com/PKSHATechnology-Research/camphr/), [transformers](https://github.com/huggingface/transformers), [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers)
- Preprocessing: [textacy](https://github.com/chartbeat-labs/textacy)
- Text Extractio: [textract (Image, Audio, PDF)](https://textract.readthedocs.io/en/stable/)
- Text Generation: [gp2client](https://github.com/rish-16/gpt2client), [textgenrnn](https://github.com/minimaxir/textgenrnn), [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple)
- Summarization: [textrank](https://github.com/summanlp/textrank), [pytldr](https://github.com/jaijuneja/PyTLDR), [bert-extractive-summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)
- Spelling Correction: [JamSpell](https://github.com/bakwc/JamSpell), [pyhunspell](https://github.com/blatinier/pyhunspell), [pyspellchecker](https://github.com/barrust/pyspellchecker), [cython_hunspell](https://github.com/MSeal/cython_hunspell), [hunspell-dictionaries](https://github.com/wooorm/dictionaries), [autocorrect (can add more languages)](https://github.com/phatpiglet/autocorrect)
- Contraction Mapping: [contractions](https://github.com/kootenpv/contractions)
- Keyword extraction: [rake](https://github.com/zelandiya/RAKE-tutorial), [pke](https://github.com/boudinfl/pke)
- Multiply Choice Question Answering: [mcQA](https://github.com/mcQA-suite/mcQA)
- Sequence to sequence models: [headliner](https://github.com/as-ideas/headliner)
- Transfer learning: [finetune](https://github.com/IndicoDataSolutions/finetune)
- Translation: [googletrans](https://pypi.org/project/googletrans/), [word2word](https://github.com/Kyubyong/word2word), [translate-python](https://github.com/terryyin/translate-python)
- Embeddings: [pymagnitude (manage vector embeddings easily)](https://github.com/plasticityai/magnitude), [chakin (download pre-trained word vectors)](https://github.com/chakki-works/chakin), [sentence-transformers](https://github.com/UKPLab/sentence-transformers), [InferSent](https://github.com/facebookresearch/InferSent), [bert-as-service](https://github.com/hanxiao/bert-as-service), [sent2vec](https://github.com/NewKnowledge/nk-sent2vec), [sense2vec](https://github.com/explosion/sense2vec), [zeugma (pretrained-word embeddings as scikit-learn transformers)](https://github.com/nkthiebaut/zeugma), [BM25Transformer](https://github.com/arosh/BM25Transformer), [laserembeddings](https://pypi.org/project/laserembeddings/)
- Multilingual support: [polyglot](https://polyglot.readthedocs.io/en/latest/index.html), [inltk (indic languages)](https://github.com/goru001/inltk), [indic_nlp](https://github.com/anoopkunchukuttan/indic_nlp_library)
- NLU: [snips-nlu](https://github.com/snipsco/snips-nlu)
- Semantic parsing: [quepy](https://github.com/machinalis/quepy)
- Inflections: [inflect](https://pypi.org/project/inflect/)
- Contractions: [pycontractions](https://pypi.org/project/pycontractions/)
- Coreference Resolution: [neuralcoref](https://github.com/huggingface/neuralcoref)
- Readability: [homer](https://github.com/wyounas/homer)
- Language Detection: [language-check](https://github.com/myint/language-check)
- Topic Modeling: [guidedlda](https://github.com/vi3k6i5/guidedlda), [enstop](https://github.com/lmcinnes/enstop)
- Clustering: [spherecluster (kmeans with cosine distance)](https://github.com/jasonlaska/spherecluster), [kneed (automatically find number of clusters from elbow curve)](https://github.com/arvkevi/kneed), [kmodes](https://github.com/nicodv/kmodes)
- Metrics: [seqeval (NER, POS tagging)](https://github.com/chakki-works/seqeval)
- String match: [jellyfish (perform string and phonetic comparison)](https://pypi.org/project/jellyfish/),[flashtext (superfast extract and replace keywords)](https://github.com/vi3k6i5/flashtext), [pythonverbalexpressions: (verbally describe regex)](https://github.com/VerbalExpressions/PythonVerbalExpressions), [commonregex (readymade regex for email/phone etc)](https://github.com/madisonmay/CommonRegex)
- Sentiment: [vaderSentiment (rule based)](https://github.com/cjhutto/vaderSentiment)
- Text distances: [textdistance](https://github.com/life4/textdistance), [editdistance](https://github.com/aflc/editdistance), [word-mover-distance](https://radimrehurek.com/gensim/models/keyedvectors.html#what-can-i-do-with-word-vectors), [wmd-relax (word mover distance for spacy)](https://github.com/src-d/wmd-relax)
- PID removal: [scrubadub](https://scrubadub.readthedocs.io/en/stable/#)
- Profanity detection: [profanity-check](https://github.com/vzhou842/profanity-check)
- Visualization: [stylecloud (wordclouds)](https://github.com/minimaxir/stylecloud), [scattertext](https://github.com/JasonKessler/scattertext)
- String Matching : [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) 
- Named Entity Recognition(NER) : [spaCy](https://spacy.io/) , [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml), [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/index.html)
- Fill blanks: [fitbert](https://github.com/Qordobacode/fitbert)
- Dictionary: [vocabulary](https://vocabulary.readthedocs.io/en/latest/usage.html)
- Nearest neighbor: [faiss](https://github.com/facebookresearch/faiss)

## Speech Recognition
- Library: [speech_recognition](https://github.com/Uberi/speech_recognition)
- Diarization: [resemblyzer](https://github.com/resemble-ai/Resemblyzer)

## RecSys
- Factorization machines (FM), and field-aware factorization machines (FFM): [xlearn](https://github.com/aksnzhy/xlearn), [DeepCTR](https://github.com/shenweichen/DeepCTR)
- Scikit-learn like API: [surprise](https://github.com/NicolasHug/Surprise)
- Recommendation System in Pytorch: [CaseRecommender](https://github.com/caserec/CaseRecommender)
- Apriori algorithm: [apyori](https://github.com/ymoch/apyori)

## Computer Vision
- Image processing: [scikit-image](https://github.com/scikit-image/scikit-image), [imutils](https://github.com/jrosebr1/imutils)
- Segmentation Models in Keras: [segmentation_models](https://github.com/qubvel/segmentation_models)
- Face recognition: [face_recognition](https://github.com/ageitgey/face_recognition), [face-alignment (find facial landmarks)](https://github.com/1adrianb/face-alignment)
- Face swapping: [faceit](https://github.com/goberoi/faceit)
- Video summarization: [videodigest](https://github.com/agermanidis/videodigest)
- Semantic search over videos: [scoper](https://github.com/RameshAditya/scoper)
- OCR: [keras-ocr](https://github.com/faustomorales/keras-ocr), [pytesseract](https://github.com/madmaze/pytesseract)
- Object detection: [luminoth](https://github.com/tryolabs/luminoth)
- Image hashing: [ImageHash](https://pypi.org/project/ImageHash/)

## Timeseries
- Predict Time Series: [prophet](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
- Scikit-learn like API: [sktime](https://github.com/alan-turing-institute/sktime)
- ARIMA models: [pmdarima](https://github.com/alkaline-ml/pmdarima)

## Framework extensions
- Pytorch: [Keras like summary for pytorch](https://github.com/sksq96/pytorch-summary), [skorch (wrap pytorch in scikit-learn compatible API)](https://github.com/skorch-dev/skorch), [catalyst](https://github.com/catalyst-team/catalyst)
- Einstein notation: [einops](https://github.com/arogozhnikov/einops)
- Scikit-learn: [scikit-lego](https://scikit-lego.readthedocs.io/en/latest/index.html), [iterstrat (cross-validation for multi-label data)](https://github.com/trent-b/iterative-stratification)
- Keras: [keras-radam](https://github.com/CyberZHG/keras-radam), [larq (binarized neural networks)](https://github.com/larq/larq), [ktrain (fastai like interface for keras)](https://pypi.org/project/ktrain/), [tavolo (useful techniques from kaggle as utilities)](https://github.com/eliorc/tavolo), [tensorboardcolab (make tensorfboard work in colab)](https://github.com/taomanwai/tensorboardcolab), [tf-sha-rnn](https://github.com/titu1994/tf-sha-rnn)

# Phase: Validation
## Model Training Monitoring
- Learning curve: [lrcurve (plot realtime learning curve in Keras)](https://github.com/AndreasMadsen/python-lrcurve), [livelossplot](https://github.com/stared/livelossplot)
- Notifications: [knockknock (get notified by slack/email)](https://github.com/huggingface/knockknock), [jupyter-notify (notify when task is completed in jupyter)](https://github.com/ShopRunner/jupyter-notify)
- Progress bar: [fastprogress](https://github.com/fastai/fastprogress)

## Interpretability
- Visualize keras models: [keras-vis](https://github.com/raghakot/keras-vis)
- Interpret models: [eli5](https://eli5.readthedocs.io/en/latest/), [lime](https://github.com/marcotcr/lime), [shap](https://github.com/slundberg/shap), [alibi](https://github.com/SeldonIO/alibi), [tf-explain](https://github.com/sicara/tf-explain), [treeinterpreter](https://github.com/andosa/treeinterpreter), [pybreakdown](https://github.com/MI2DataLab/pyBreakDown), [xai](https://github.com/EthicalML/xai), [lofo-importance](https://github.com/aerdem4/lofo-importance)
- Interpret BERT: [exbert](http://exbert.net/exBERT.html?sentence=I%20liked%20the%20music&layer=0&heads=..0,1,2,3,4,5,6,7,8,9,10,11&threshold=0.7&tokenInd=null&tokenSide=null&maskInds=..9&metaMatch=pos&metaMax=pos&displayInspector=null&offsetIdxs=..-1,0,1&hideClsSep=true)
- Interpret word2vec: [word2viz](https://lamyiowce.github.io/word2viz/)

# Phase: Optimization
## Hyperparameter Optimization
- Keras: [keras-tuner](https://github.com/keras-team/keras-tuner)
- Scikit-learn: [sklearn-deap (evolutionary algorithm for hyperparameter search)](https://github.com/rsteca/sklearn-deap), [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
- General: [hyperopt](https://github.com/hyperopt/hyperopt), [optuna](https://optuna.org/), [evol](https://github.com/godatadriven/evol), [talos](https://github.com/autonomio/talos)

## Visualization
- Draw CNN figures: [nn-svg](http://alexlenail.me/NN-SVG/LeNet.html)
- Visualization for scikit-learn: [yellowbrick](https://www.scikit-yb.org/en/latest/index.html), [scikit-plot](https://scikit-plot.readthedocs.io/en/stable/metrics.html)
- XKCD like charts: [chart.xkcd](https://timqian.com/chart.xkcd/)
- Convert matplotlib charts to D3 charts: [mpld3](http://mpld3.github.io/index.html)
- Generate graphs using markdown: [mermaid](https://mermaid-js.github.io/mermaid/#/README)
- Visualize topics models: [pyldavis](https://pyldavis.readthedocs.io/en/latest/)
- High dimensional visualization: [umap](https://github.com/lmcinnes/umap)
- Visualization libraries: [pygal](http://www.pygal.org/en/latest/index.html), [plotly](https://github.com/plotly/plotly.py), [plotnine](https://github.com/has2k1/plotnine)
- Interactive charts: [bokeh](https://github.com/bokeh/bokeh)
- Visualize architectures: [netron](https://github.com/lutzroeder/netron)
- Activation maps for keras: [keract](https://github.com/philipperemy/keract)
- Create interactive charts online: [flourish-studio](https://flourish.studio/)
- Color Schemes: [open-color](https://yeun.github.io/open-color/)

# Phase: Production
## Model Serialization
- Transpiling: [sklearn-porter (transpile sklearn model to C, Java, JavaScript and others)](https://github.com/nok/sklearn-porter), [m2cgen](https://github.com/BayesWitnesses/m2cgen)
- Pickling extended: [cloudpickle](https://github.com/cloudpipe/cloudpickle), [jsonpickle](https://github.com/jsonpickle/jsonpickle)

## Scalability
- Parallelize Pandas: [pandarallel](https://github.com/nalepae/pandarallel), [swifter](https://github.com/jmcarpenter2/swifter), [modin](https://github.com/modin-project/modin)
- Parallelize numpy operations: [numba](http://numba.pydata.org/)

## Bechmark
- Profile pytorch layers: [torchprof](https://github.com/awwong1/torchprof)

## API
- Configuration Management: [config](https://pypi.org/project/config/), [python-decouple](https://github.com/henriquebastos/python-decouple)
- Data Validation: [schema](https://github.com/keleshev/schema), [jsonschema](https://pypi.org/project/jsonschema/), [cerebrus](https://github.com/pyeve/cerberus), [pydantic](https://pydantic-docs.helpmanual.io/), [marshmallow](https://marshmallow.readthedocs.io/en/stable/), [validators](https://validators.readthedocs.io/en/latest/#basic-validators)
- Enable CORS in Flask: [flask-cors](https://flask-cors.readthedocs.io/en/latest/)
- Caching: [cachetools](https://pypi.org/project/cachetools/), [cachew (cache to local sqlite)](https://github.com/karlicoss/cachew)
- Authentication: [pyjwt (JWT)](https://github.com/jpadilla/pyjwt)
- Task Queue: [rq](https://github.com/rq/rq), [schedule](https://github.com/dbader/schedule)
- Database: [flask-sqlalchemy](https://github.com/pallets/flask-sqlalchemy), [tinydb](https://github.com/msiemens/tinydb)
- Logging: [loguru](https://github.com/Delgan/loguru)

## Dashboard
- Generate frontend with python: [streamlit](https://github.com/streamlit/streamlit)

## Adversarial testing
- Generate images to fool model: [foolbox](https://github.com/bethgelab/foolbox)
- Generate phrases to fool NLP models: [triggers](https://www.ericswallace.com/triggers)
- General: [cleverhans](https://github.com/tensorflow/cleverhans)

## Python libraries
- Datetime compatible API for Bikram Sambat: [nepali-date](https://github.com/arneec/nepali-date)
- bloom filter: [python-bloomfilter](https://github.com/jaybaird/python-bloomfilter)
- Run python libraries in sandbox: [pipx](https://github.com/pipxproject/pipx)
- Pretty print tables in CLI: [tabulate](https://pypi.org/project/tabulate/)
- Leaflet maps from python: [folium](https://python-visualization.github.io/folium/)
- Debugging: [PySnooper](https://github.com/cool-RR/PySnooper)
- Date and Time: [pendulum](https://github.com/sdispater/pendulum)
- Create interactive prompts: [prompt-toolkit](https://pypi.org/project/prompt-toolkit/)
- Concurrent database: [pickleshare](https://pypi.org/project/pickleshare/)
