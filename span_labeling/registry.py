# from span_labeling.dataset import ErrorDataset, NerDataset, SyntheticDataset
# from span_labeling.methods.json_method import JSONSpanLabeler
# from span_labeling.methods.xml_method import XMLSpanLabeler
# from span_labeling.methods.index_method import IndexSpanLabeler
# from span_labeling.methods.occurrence_method import JSONOccurrenceSpanLabeler




# TASKS = {
#     'error': ErrorDataset(path="data/custom/error_test.json"),
#     'ner': NerDataset(path="data/custom/ner_test.json"),
#     'synthetic': SyntheticDataset(path="data/custom/synthetic_test.json")
# }

# METHODS = {
#     'json': JSONSpanLabeler(model_name="hermes3:8b"),
#     'xml': XMLSpanLabeler(model_name="hermes3:8b"),
#     'index': IndexSpanLabeler(model_name="hermes3:8b"),
#     'occurrence': JSONOccurrenceSpanLabeler(model_name="hermes3:8b")
# }


EXAMPLES ={
        "synthetic" :
    {
        # "json":
        "occurrence" : """
Example:
Task: Find all sequences matching 'schwarz laufen' that are not preceded by 'gelb'.
Text: schiff flugzeug licht wasser schwarz laufen ruhig wasser flugzeug licht mögen flugzeug licht flugzeug licht laufen gebäude aufgeregt schwarz flugzeug licht wasser ruhig wasser wasser
JSON Output: [{"text": "schwarz laufen", "occurrence": 1}]
"""

    }





}