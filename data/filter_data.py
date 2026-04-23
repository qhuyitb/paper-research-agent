from tqdm import tqdm
import json
import os

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL"}

def filter_arxiv_papers(input_path="data/raw/arxiv-metadata-oai-snapshot.json", 
                         save_path="data/raw/papers.json", 
                         max_papers=20000):
    papers = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning papers"):
            line = line.strip()
            if not line:
                continue

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue  
            # Filter theo category
            categories = set(paper.get("categories", "").split())
            year = paper.get("update_date", "")[:4]
            if year < "2018":  
                continue
            if categories & TARGET_CATEGORIES:
                papers.append({
                    "id": paper.get("id", ""),
                    "title": paper.get("title", "").strip(),
                    "abstract": paper.get("abstract", "").strip(),
                    "authors": paper.get("authors", ""),
                    "categories": paper.get("categories", ""),
                    "year": paper.get("update_date", "")[:4],
                    "url": f"https://arxiv.org/abs/{paper.get('id', '')}",
                })

            if len(papers) >= max_papers:
                break

    # Lưu ra file JSON
    os.makedirs("data/raw", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(papers)} papers vào {save_path}")
    return papers


if __name__ == "__main__":
    papers = filter_arxiv_papers()

    print("\nPreview 2 papers đầu tiên:")
    for p in papers[:2]:
        print(f"\nTitle: {p['title']}")
        print(f"Year: {p['year']}")
        print(f"Categories: {p['categories']}")
        print(f"Abstract (100 chars): {p['abstract'][:100]}...")