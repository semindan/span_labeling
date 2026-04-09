from span_labeling.config import DatasetConfig

NER_ALL = [
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_ceb_gja.json", name="uner_ceb_gja"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_da_ddt.json", name="uner_da_ddt"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_de_pud.json", name="uner_de_pud"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_en_ewt.json", name="uner_en_ewt"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_en_pud.json", name="uner_en_pud"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_hr_set.json", name="uner_hr_set"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_pt_bosque.json", name="uner_pt_bosque"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_pt_pud.json", name="uner_pt_pud"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_ru_pud.json", name="uner_ru_pud"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_sk_snk.json", name="uner_sk_snk"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_sr_set.json", name="uner_sr_set"
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_sv_pud.json", name="uner_sv_pud"
    ),
    DatasetConfig(
        type="ner",
        path="data/universal_ner/uner_sv_talbanken.json",
        name="uner_sv_talbanken",
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_tl_trg.json", name="uner_tl_trg"
    ),
    DatasetConfig(
        type="ner",
        path="data/universal_ner/uner_tl_ugnayan.json",
        name="uner_tl_ugnayan",
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_zh_gsd.json", name="uner_zh_gsd"
    ),
    DatasetConfig(
        type="ner",
        path="data/universal_ner/uner_zh_gsdsimp.json",
        name="uner_zh_gsdsimp",
    ),
    DatasetConfig(
        type="ner", path="data/universal_ner/uner_zh_pud.json", name="uner_zh_pud"
    ),
]

WMT_NEWS = [
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-cs-news.json", name="wmt-en-cs-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-es-news.json", name="wmt-en-es-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-hi-news.json", name="wmt-en-hi-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-is-news.json", name="wmt-en-is-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ja-news.json", name="wmt-en-ja-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ru-news.json", name="wmt-en-ru-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-uk-news.json", name="wmt-en-uk-news"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-zh-news.json", name="wmt-en-zh-news"
    ),
]

WMT_LITERARY = [
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-cs-literary.json", name="wmt-en-cs-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-es-literary.json", name="wmt-en-es-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-hi-literary.json", name="wmt-en-hi-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-is-literary.json", name="wmt-en-is-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ja-literary.json", name="wmt-en-ja-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ru-literary.json", name="wmt-en-ru-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-uk-literary.json", name="wmt-en-uk-literary"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-zh-literary.json", name="wmt-en-zh-literary"
    ),
]

WMT_SOCIAL = [
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-cs-social.json", name="wmt-en-cs-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-es-social.json", name="wmt-en-es-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-hi-social.json", name="wmt-en-hi-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-is-social.json", name="wmt-en-is-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ja-social.json", name="wmt-en-ja-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-ru-social.json", name="wmt-en-ru-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-uk-social.json", name="wmt-en-uk-social"
    ),
    DatasetConfig(
        type="wmt", path="data/wmt/wmt-en-zh-social.json", name="wmt-en-zh-social"
    ),
]

SYNTHETIC_ALL = [
    DatasetConfig(
        type="synthetic",
        path="data/synthetic/english_word_synthetic_data.json",
        name="english_word_synthetic_data",
    ),
    DatasetConfig(
        type="synthetic",
        path="data/synthetic/english_non_overlapping_word_synthetic_data.json",
        name="english_non_overlapping_word_synthetic_data",
    ),
]

MULTIGEC_ALL = [
    DatasetConfig(
        type="multigec", path="data/multigec/multigec_en.json", name="multigec_en"
    ),
]

WMT_ALL = WMT_NEWS + WMT_LITERARY + WMT_SOCIAL

ALL = NER_ALL + WMT_ALL + SYNTHETIC_ALL + MULTIGEC_ALL

DATASET_GROUPS: dict[str, list[DatasetConfig]] = {
    "ner_all": NER_ALL,
    "wmt_news": WMT_NEWS,
    "wmt_literary": WMT_LITERARY,
    "wmt_social": WMT_SOCIAL,
    "wmt_all": WMT_ALL,
    "synthetic_all": SYNTHETIC_ALL,
    "multigec_all": MULTIGEC_ALL,
    "all": ALL,
}
