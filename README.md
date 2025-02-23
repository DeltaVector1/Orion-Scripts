1. Pruning Unnecessary Fields (1.py):
   - The initial script `1.py` prunes the JSON records to include only the fields "id", "title", and "content".
2. Language Filtering (2.py):
   - The script 2.py filters the dataset to keep only records with English content using the langdetect lib
3. Tokenization and Length Filtering (3.py):
   - The script 3.py uses HF tokenizers to tokenize the "content" field using a specific model (e.g., "microsoft/phi-4"). 
   - It filters out records that exceed a predefined maximum token limit (e.g., 16384)  
4. Deduplication (4.py):
   - The script `4.py` deduplicates the dataset based on the "content" field. 
   - It ensures that each record with a unique "content" value is retained.
5. Fuzzy Deduplication (5.py):
   - The script `5.py` performs fuzzy deduplication using the rapidfuzz lib
   - It checks for similar "content" values within a certain threshold (e.g., 85% similarity) and removes duplicates.
6. Content Rating (6.py):
   - The script 6.py sends each "content" for evaluation based on its writing quality using Supernova-Medius from 1-5 (Though Medius ended up rating a few stories as 6???)
   - Ratings were cut short due the evals taking too long (5~ Days), I ended up with a 35K subset of which 16K stories were extracted from. Although I plan to perform a larger subset in the future. 
7. Filtering Based on Rating (Extract.py):
   - The script `Extract.py` filters the rated JSON file to retain records with specific rating criteria (e.g., 4 to 6).
