import Levenshtein

def compute_exact_match(predictions, references):
    """
    Computes Exact Match (EM) score.
    Returns the percentage of predictions that exactly match the reference.
    """
    if not predictions or not references:
        return 0.0
    
    matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
    return matches / len(predictions)

def compute_anls(predictions, references, threshold=0.5):
    """
    Computes Average Normalized Levenshtein Similarity (ANLS).
    ANLS = 1 - (Levenshtein_distance / max(len(p), len(r)))
    If ANLS < threshold, score is 0.
    """
    if not predictions or not references:
        return 0.0
        
    total_score = 0.0
    for p, r in zip(predictions, references):
        p, r = p.strip().lower(), r.strip().lower()
        if not p and not r:
            total_score += 1.0
            continue
            
        dist = Levenshtein.distance(p, r)
        max_len = max(len(p), len(r))
        
        if max_len == 0:
            score = 1.0
        else:
            similarity = 1.0 - (dist / max_len)
            score = similarity if similarity >= threshold else 0.0
            
        total_score += score
        
    return total_score / len(predictions)
