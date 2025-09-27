# LFGPT-data: Dataset for Hallucination-Aware Large Language Model in Swine Farming


## Project Overview

LFGPT-data is a comprehensive dataset repository supporting research on hallucination detection and mitigation in agricultural large language models. This repository contains high-quality domain-specific data tailored for swine farming applications.

## Dataset Description

### 1. Hallucination\_data

A benchmark dataset for detecting and classifying domain-specific hallucinations in swine farming.

#### Key Features:



* **Total Samples**: 10,383 labeled samples

* **Hallucination Taxonomy**: 10 subtypes across 2 major categories

* **Distribution**:


  * Factual Hallucinations: 62.7%

  * Fidelity Hallucinations: 37.3%

#### Hallucination Categories and Subtypes:

**Factual Hallucinations (62.7%)**:



* **Concept Confusion** (17.36%): Misunderstanding of domain-specific concepts

* **Numerical Error** (15.24%): Incorrect numbers, units, or measurements

* **Temporal Inconsistency** (12.89%): Time-related factual errors

* **Spatial Misinformation** (8.91%): Location or space-related inaccuracies

* **Causal Fallacy** (8.30%): Incorrect cause-effect relationships

**Fidelity Hallucinations (37.3%)**:



* **Logical Contradiction** (12.45%): Internal logical inconsistencies

* **Contextual Irrelevance** (9.87%): Content not relevant to the question context

* **Incomplete Information** (7.63%): Missing critical details

* **Redundancy** (4.21%): Excessive repetitive content

* **Ambiguity** (3.14%): Unclear or vague statements

### 2. High-quality Q\&A

A curated dataset of question-answer pairs for training domain-specific LLMs.

#### Key Features:



* **Total Pairs**: 62,232 high-quality QA pairs

* **Domain Coverage**: Full swine production lifecycle

* **Expert Validation**: All pairs verified by domain experts

#### Knowledge Domains:



* **Veterinary Care**: Disease diagnosis, treatment protocols, health management

* **Nutrition Planning**: Feed formulation, dietary requirements, nutritional management

* **Reproduction**: Breeding cycles, mating strategies, farrowing management

* **Environmental Management**: Housing conditions, temperature control, waste management

* **Biosecurity**: Disease prevention, hygiene protocols, biocontainment measures

## Data Collection Methodology

### Hallucination Dataset Construction



1. **Base QA Generation**: Using DeepSeek-Distill-Qwen32B model with prompt engineering

2. **Hallucination Injection**: Controlled generation of errors using DeepSeek-v3

3. **Expert Annotation**: Manual verification and classification by domain experts

4. **Taxonomy Development**: Creation of swine farming-specific hallucination categories

### High-Quality QA Corpus Construction



1. **Data Sourcing**: Collection from agricultural publications, veterinary manuals, and expert knowledge

2. **QA Pair Generation**: Using state-of-the-art LLMs with domain-specific prompts

3. **Expert Validation**:

* Domain accuracy verification

* Terminological precision checks

* Logical completeness validation

1. **Quality Assurance**: Multiple rounds of review and refinement

## Data Format

### Hallucination\_data



```
{
  "hallucination_type": "Factual Hallucination",
  "sub_type": "Data Distortion (Numerical/Proportion Bias)",
  "hallucinated_case": "Indirect hemagglutination test: titer >1:5 is positive, titer <1:2 is negative.",
  "reference_case": "Indirect hemagglutination test: titer >1:10 is positive, titer <1:5 is negative, and between 1:5 and 1:10 is suspicious. Negative and suspicious cases should be retested after 4 weeks. Two consecutive negative results indicate no enzootic pneumonia, while two consecutive suspicious results indicate positivity.",
  "error_detail": "Incorrectly adjusted the judgment criteria for the indirect hemagglutination test, which may lead to inaccurate diagnostic results.",
  "original_question": "How to diagnose Mycoplasma hyopneumoniae in pigs?",
  "original_input": ""
}
```

### High-quality Q\&A



```
{
  "instruction": "What are the effects of microbial agents on pig farming?",
  "input": "",
  "output": "Microbial agents can significantly increase the weight gain rate of pigs (49.92%), while reducing diarrhea rate and mortality rate, and improving the mental state and appearance quality of the pig herd." 
}
```

## Usage Instructions

### Requirements



* Python 3.8+

* pandas 1.5+

* numpy 1.23+

### Loading the Dataset



```
import pandas as pd

\# Load hallucination dataset

hallucination_files = [
    './Hallucination\_data/1.1 Data Distortion.json',
    './Hallucination\_data/1.2 Concept Confusion.json',
    # Other hallucination data files
]

hallucination_dfs = []
for file in hallucination_files:
    df = pd.read_json(file)
    hallucination_dfs.append(df)

hallucination_df = pd.concat(hallucination_dfs, ignore_index=True)

\# Load high-quality QA dataset

qa_df = pd.read_json('./High-quality Q\&A/Pig\_QA.json')

print(f"Hallucination dataset size: {len(hallucination_df)} samples")

print(f"QA dataset size: {len(qa_df)} pairs")
```

### Data Preprocessing



```
def preprocess_text(text):
    """Basic text preprocessing for LLM training"""
    text = text.strip()
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

\# Apply preprocessing to hallucination data
hallucination_df['processed_question'] = hallucination_df['original_question'].apply(preprocess_text)
hallucination_df['processed_hallucination'] = hallucination_df['hallucinated_case'].apply(preprocess_text)
hallucination_df['processed_reference'] = hallucination_df['reference_case'].apply(preprocess_text)

\# Apply preprocessing to high-quality QA data
qa_df['processed_instruction'] = qa_df['instruction'].apply(preprocess_text)
qa_df['processed_output'] = qa_df['output'].apply(preprocess_text)
```

## Research Applications

### 1. Hallucination Detection Research



* Train and evaluate hallucination detection models

* Develop domain-specific hallucination taxonomies

* Study error patterns in agricultural LLMs

### 2. Domain Adaptation



* Fine-tune LLMs for swine farming applications

* Develop specialized agricultural AI systems

* Improve model performance in domain-specific tasks

### 3. Trustworthy AI in Agriculture



* Build reliable decision support systems

* Enhance food safety and animal welfare

* Support sustainable farming practices



## Contact Information

For questions or issues regarding the dataset, please contact:

* Email: \[yliu6792@gmail.com]

* GitHub Issues: [GitHub Issues Page](https://github.com/Blueeeecho/LFGPT-data/issues)

## Changelog

### Version 1.0 (2024-09-22)



* Initial release of Hallucination\_data (10,383 samples)

* Initial release of High-quality Q\&A (62,232 pairs)

* Complete documentation and usage examples



***

*This dataset supports the development of trustworthy AI systems in agriculture, with a focus on reducing hallucinations and improving decision reliability in livestock farming.*
