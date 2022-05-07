
from collections import namedtuple
import jsonlines
from typing import Iterable, List
import xml.etree.ElementTree as ET


TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

def load_sick(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_SICK = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
    NLI_LABEL_MAP_SICK = {k: i for i, k in enumerate(NLI_LABELS_SICK)}
    
    filename="/users7/hcxu/project/dataset/SICK/SICK.txt"
    with open(filename) as f:
            f.readline()
            lines = f.readlines()
    out = []
    for line in lines:
        line = line.split("\t")
        
        out.append(
            TextPairExample(line[0], line[1], line[2], NLI_LABEL_MAP_SICK[line[3]]))
    return out

#load_sick(False)

def load_anli_r1(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_anli = ["c", "e", "n"]
    NLI_LABEL_MAP_anli = {k: i for i, k in enumerate(NLI_LABELS_anli)}
    
    num=1
    filename_anli_r1="/users7/hcxu/project/dataset/anli_v1.0/R1/test.jsonl"
    out = []
    with jsonlines.open(filename_anli_r1) as reader:
        for item in reader:
            #if item['hypothesis']=="Another individual laid waste to Roberto Javier Mora Garcia.":
            #    print(item)

            out.append(
                TextPairExample(str(num), item['context'], item['hypothesis'], NLI_LABEL_MAP_anli[item['label']]))
            num=num+1
    
    return out


def load_anli_r2(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_anli = ["c", "e", "n"]
    NLI_LABEL_MAP_anli = {k: i for i, k in enumerate(NLI_LABELS_anli)}
    
    num=1
    filename_anli_r2="/users7/hcxu/project/dataset/anli_v1.0/R2/test.jsonl"
    out = []
    with jsonlines.open(filename_anli_r2) as reader:
        for item in reader:
            out.append(
                TextPairExample(str(num), item['context'], item['hypothesis'], NLI_LABEL_MAP_anli[item['label']]))
            num=num+1
    
    return out
    

def load_anli_r3(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_anli = ["c", "e", "n"]
    NLI_LABEL_MAP_anli = {k: i for i, k in enumerate(NLI_LABELS_anli)}
    
    num=1
    filename_anli_r3="/users7/hcxu/project/dataset/anli_v1.0/R3/test.jsonl"
    out = []
    with jsonlines.open(filename_anli_r3) as reader:
        for item in reader:
            out.append(
                TextPairExample(str(num), item['context'], item['hypothesis'], NLI_LABEL_MAP_anli[item['label']]))
            num=num+1
    
    return out


def load_scitail(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_scitail = ["CONTRADICTION", "entails", "neutral"]
    NLI_LABEL_MAP_scitail = {k: i for i, k in enumerate(NLI_LABELS_scitail)}
    
    filename="/users7/hcxu/project/dataset/scitail/tsv_format/scitail_1.0_test.tsv"

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    out = []
    num=1
    for line in lines:
        line = line.split("\t")       
        out.append(
            TextPairExample(str(num), line[0], line[1], NLI_LABEL_MAP_scitail[line[2].strip()]))
        num=num+1
    return out


def load_RTE1(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_RTE = ["NO", "YES", "UNKNOWN"]
    NLI_LABEL_MAP_RTE = {k: i for i, k in enumerate(NLI_LABELS_RTE)}
    
    num=1
    out = []
    tree1 = ET.parse('/users7/hcxu/project/dataset/RTE/RTE1_test_3ways.xml')
    root1 = tree1.getroot()
    for pair in root1.findall('pair'): #element.findall()
        label=pair.get('entailment')  
        premise=pair.find('t').text  
        hypothsis=pair.find('h').text  
       
        out.append(
            TextPairExample(str(num), premise, hypothsis, NLI_LABEL_MAP_RTE[label]))
        num=num+1

    tree2 = ET.parse('/users7/hcxu/project/dataset/RTE/RTE2_test_3ways.xml')
    root2 = tree2.getroot()
    for pair in root2.findall('pair'): #element.findall()
        label=pair.get('entailment')  
        premise=pair.find('t').text  
        hypothsis=pair.find('h').text  
       
        out.append(
            TextPairExample(str(num), premise, hypothsis, NLI_LABEL_MAP_RTE[label]))
        num=num+1

    tree3 = ET.parse('/users7/hcxu/project/dataset/RTE/RTE3_test_3ways.xml')
    root3 = tree3.getroot()
    for pair in root3.findall('pair'): #element.findall()
        label=pair.get('entailment')  
        premise=pair.find('t').text  
        hypothsis=pair.find('h').text  
        
        out.append(
            TextPairExample(str(num), premise, hypothsis, NLI_LABEL_MAP_RTE[label]))
        num=num+1
    return out    


def load_glue(is_train, sample=None, custom_path=None) -> List[TextPairExample]:

    NLI_LABELS_GLUE = ["contradiction", "entailment", "neutral"]
    NLI_LABEL_MAP_GLUE = {k: i for i, k in enumerate(NLI_LABELS_GLUE)}
    
    filename="/users7/hcxu/project/dataset/GLUE/diagnostic-full.tsv"

    with open(filename, 'r', encoding='utf-8') as f:
            f.readline()
            lines = f.readlines()
    out = []
    num=1
    for line in lines:
        line = line.split("\t")
        # print(line[5])
        # print(line[6])
        # print(NLI_LABEL_MAP_GLUE[line[7].strip()])
        out.append(
            TextPairExample(str(num), line[5], line[6], NLI_LABEL_MAP_GLUE[line[7].strip()]))
        num=num+1
        
    return out   
