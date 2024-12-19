# Exploring the Possibility of Predicting Intelligence and Personality Traits of an Individual Using Transcript of Their Speech

Aryan Ahadinia, Saba Nasiri, Shahrzad Javidi

Intelligence and personality have always been considered two important dimensions for characterizing an individual. Recent advancements in Natural Language Processing methods have initiated an interdisciplinary field of research aimed at inferring these variables. This report explores the possibility of inferring such variables. In this project, the goal is to investigate this potential using a new dataset (IDIAP). Our studies shed light on the possibility of accurately inferring intelligence and, to some extent, personality.

## How to Run?

### Chunking

```bash
python3 run_chunking_aug.py -i <input_file> -o <output_file>
```

### Embedding Generation

Proceed to the notebook `notebooks/embedding_generation.ipynb` and run the cells.

### Feature Extraction

```bash
python run_feature_extraction.py --type my_personality | idiap | idiap_chunked --datapath <path_to_data> --output_path_NRC <path_to_output> --output_path_NRC_VAD <path_to_output> --output_path_readability <path_to_output> --output_path_LIWC <path_to_output> --output_path <path_to_output> --target_path <path_to_output>
```

### Chunking Augmentation

```bash
python run_chunking_aug.py  -i <input_file> -o <output_file>
```

### Running SVR

```bash
python run_svr.py --type my_personality | idiap | idiap_chunked --datapath_features <path_to_data> --datapath_targets .<path_to_targets> --target <name_of_the_target_col> --features embeddings | psycological | combined -- datapath_features2 <only_for_combined>
```

### GoEmotion Model

proceed to the notebook `notebooks/emotion_extraction` and run the cells.

```bash
python run_light_gbm.py --type my_personality | idiap | idiap_chunked --datapath_features <path_to_data> --datapath_targets .<path_to_targets> --target <name_of_the_target_col> --features embeddings | psycological | combined -- datapath_features2 <only_for_combined>
```
