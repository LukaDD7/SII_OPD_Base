# 数学题 verifier: 检查 answer 是否匹配 ground truth
def check(answer, ground_truth):
    try:
        return float(answer.strip()) == float(ground_truth.strip())
    except:
        return answer.strip() == ground_truth.strip()
