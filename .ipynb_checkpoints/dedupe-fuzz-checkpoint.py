from rapidfuzz import fuzz, process
import orjson
from multiprocessing import Pool, Manager
from tqdm import tqdm

input_file = "Text.jsonl"
output_file = "filtered_file.jsonl"
similarity_threshold = 85  # Percentage threshold for similarity
num_workers = 15  # Use your available cores
batch_size = 1000  # Number of records per chunk


def is_similar(new_content, seen_contents):
    """
    Check for similarity to already-seen contents using RapidFuzz.
    """
    matches = process.extract(
        new_content, seen_contents, scorer=fuzz.ratio, limit=1
    )  # Check against limited candidates
    if matches and matches[0][1] >= similarity_threshold:
        return True
    return False


def process_chunk(chunk, shared_seen_contents, lock, chunk_id=0):
    """
    Deduplicate a chunk of records.
    """
    local_seen = set()  # A local set to avoid duplicates within this chunk
    unique_records = []  # List of unique records to return
    skipped_records = 0  # Counter for skipped records

    for line in tqdm(chunk, desc=f"Chunk {chunk_id}", leave=False):
        try:
            record = orjson.loads(line)
            content = record.get("text", "")

            if not content:
                # Skip records with empty content
                skipped_records += 1
                continue

            with lock:
                if content in shared_seen_contents:
                    # Already globally seen; skip this record
                    skipped_records += 1
                    continue

            # Perform fuzzy matching locally
            if not is_similar(content, local_seen):
                local_seen.add(content)
                unique_records.append(record)
            else:
                # Fuzzy match too similar; skip record
                skipped_records += 1
        except Exception as e:
            print(f"Error processing record: {e}")
            skipped_records += 1

    with lock:
        # Fixed update mechanism for ListProxy object
        for item in local_seen:
            if item not in shared_seen_contents:
                shared_seen_contents.append(item)

    print(f"Chunk {chunk_id} processed. Unique records: {len(unique_records)}, Skipped records: {skipped_records}")
    return unique_records


def main():
    # Count total lines for better progress tracking
    total_lines = 0
    with open(input_file, "r") as infile:
        for _ in tqdm(infile, desc="Counting lines"):
            total_lines += 1
            
    print(f"Total lines in input file: {total_lines}")
    
    # Read file in chunks for streaming
    chunks = []
    with open(input_file, "r") as infile:
        current_chunk = []
        for i, line in enumerate(tqdm(infile, desc="Creating chunks", total=total_lines)):
            current_chunk.append(line)
            if len(current_chunk) >= batch_size:
                chunks.append(current_chunk)
                current_chunk = []
                
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
    print(f"Created {len(chunks)} chunks for processing")

    # Set up shared memory using Manager
    manager = Manager()
    shared_seen_contents = manager.list()  # Shared content tracker
    lock = manager.Lock()

    # Use multiprocessing to process each chunk
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_chunk,
                    [(chunk, shared_seen_contents, lock, i) for i, chunk in enumerate(chunks)],
                ),
                desc="Multiprocessing fuzzy deduplication",
                total=len(chunks),
            )
        )

    # Flatten all the unique records from the multiprocessing results
    filtered_records = [record for chunk_results in results for record in chunk_results]

    print(f"Total unique records after processing: {len(filtered_records)}")

    # Write the deduplicated records to the output file
    with open(output_file, "w") as outfile:
        for record in tqdm(filtered_records, desc="Writing output"):
            outfile.write(orjson.dumps(record).decode("utf-8") + "\n")


if __name__ == "__main__":
    main()