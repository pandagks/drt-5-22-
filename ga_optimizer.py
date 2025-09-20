import random
from statistics import mean, stdev
from utils import get_distance_between

# ---------------------------
# Helper Functions
# ---------------------------

def _safe_dist(a, b):
    d = get_distance_between(a, b)
    return d if isinstance(d, (int, float)) and d is not None else 0.0

def generate_valid_sequence(pairs):
    seen = set() # 중복 제거용 집합
    pickups = [] # 승차 정류장만 저장
    for p, _ in pairs:
        if p not in seen:
            pickups.append(p)
            seen.add(p)
    dropoffs = [] # 하차 정류장만 저장
    for _, d in pairs:
        if d not in seen:
            dropoffs.append(d)
            seen.add(d)
    sequence = pickups.copy()
    remaining = dropoffs.copy()
    random.shuffle(remaining)
    # 제약조건 1. 승차 후 하차 순서 보장장\
    for drop in remaining:
        idx = random.randint(0, len(sequence))
        while sequence.index([p for p, d in pairs if d == drop][0]) >= idx:
            idx += 1
        sequence.insert(idx, drop)
    return sequence

#최적화 목적 함수 중복 제거된 순서대로 정류장을 순회하며 거리 합계를 계산 
def evaluate_sequence(seq):
    seen = set()
    unique_seq = []
    for stop in seq:
        if stop not in seen:
            unique_seq.append(stop) # 중복 정류장 제거 
            seen.add(stop)
    seq = unique_seq
    total = 0 
    for i in range(len(seq) - 1):
        dist = get_distance_between(seq[i], seq[i+1])
        total += dist if dist else 0
    return total

#초기 개체군 생성
def initialize_population(pairs, size=50):
    return [generate_valid_sequence(pairs) for _ in range(size)]

# parent1 앞쪽 자르고, parent2에서 중복 제거 후 이어 붙이기
def crossover(parent1, parent2):
    if len(parent1) < 3:
        return parent1[:]  # 너무 짧으면 복사만
    cut = random.randint(1, len(parent1) - 2)
    head = parent1[:cut] 
    tail = [x for x in parent2 if x not in head]
    return head + tail

# 픽업 정류장은 돌연변이 대상에서 제외
def mutate(seq, pickup_set):
    idx1, idx2 = random.sample(range(len(seq)), 2)
    if seq[idx1] not in pickup_set and seq[idx2] not in pickup_set:
        seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
    return seq

# ---------------------------
# Main GA Function
# ---------------------------
total_distance_across_runs = 0
total_time_across_runs = 0

# 유전 알고리즘
def run_ga(pairs, generations=100, pop_size=100, verbose=True, plot=False):
    global total_distance_across_runs, total_time_across_runs

    # 빈 입력 방어
    if not pairs:
        return [], [], 0.0, 0

    # 1) 초기화
    population = initialize_population(pairs, pop_size)
    pickup_set = set(p for p, _ in pairs)
    fitness_history = []

    # 2) 세대 반복
    for gen in range(generations):
        # 개체 평가
        scored = [(evaluate_sequence(ind), ind) for ind in population if ind]
        if not scored:
            # 전부 빈 경로라면 재초기화
            population = initialize_population(pairs, pop_size)
            scored = [(evaluate_sequence(ind), ind) for ind in population]

        scored.sort(key=lambda x: x[0])
        best_score, best_seq = scored[0]

        # 복귀 포함 피트니스 (여기서 population/ scored가 이미 존재)
        fitness_with_return = [
            evaluate_sequence(path) + _safe_dist(path[-1], "00_오이도차고지")
            for _, path in scored
        ]

        fitness_history.append(best_score)

        # 다음 세대 생성 (elitism + crossover + mutate)
        next_gen = [best_seq]
        # 상위 20개 또는 가능한 만큼에서 부모 선택
        parent_pool = scored[:min(20, len(scored))]
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(parent_pool, 2)
            child = crossover(p1[1], p2[1])
            child = mutate(child, pickup_set)
            next_gen.append(child)

        population = next_gen

    # 3) 최종 베스트 경로 산출
    if not population:
        return [], [], 0.0, 0

    best = min(population, key=evaluate_sequence)
    # 중복 정류장 제거 (앞선 최초 등장만 유지)
    dedup = []
    seen = set()
    for s in best:
        if s not in seen:
            dedup.append(s); seen.add(s)
    best = dedup

    # 4) 총 거리/시간 계산 (None 안전 처리)
    total_distance = 0.0
    total_minutes = 0
    for i in range(len(best) - 1):
        dist = _safe_dist(best[i], best[i+1])
        if dist > 0:
            minutes = int(dist * 3)
            total_distance += dist
            total_minutes += minutes
            if verbose:
                print(f"  {best[i]} -> {best[i+1]} : {dist:.2f} km / {minutes}분")

    # 차고지 복귀
    if best:
        last_stop = best[-1]
        return_to_depot = _safe_dist(last_stop, "00_오이도차고지")
        if return_to_depot > 0:
            minutes_back = int(return_to_depot * 3)
            if verbose:
                print(f"  {last_stop} -> 00_오이도차고지 : {return_to_depot:.2f} km / {minutes_back}분 (복귀)")
            total_distance += return_to_depot
            total_minutes += minutes_back

    # 누적 갱신
    total_distance_across_runs += total_distance
    total_time_across_runs += total_minutes

    if verbose:
        print(f"[GA] 총 이동 거리: {total_distance:.2f} km")
        print(f"[GA] 총 예상 소요 시간: {total_minutes}분")
        print(f"[GA 누적] 전체 시간대 총 이동 거리 합: {total_distance_across_runs:.2f} km")
        print(f"[GA 누적] 전체 시간대 총 소요 시간 합: {total_time_across_runs}분")
        print("[GA 최종 요약]")
        print(f"총 누적 거리: {total_distance_across_runs:.2f} km")
        print(f"총 누적 시간: {total_time_across_runs}분")

    # 마지막 세대의 복귀 포함 피트니스도 돌려줘야 하면, 위에서 계산한 값을 재사용/반환
    # (필요 없다면 빈 리스트 반환해도 됨)
    fitness_with_return = [
        evaluate_sequence(path) + _safe_dist(path[-1], "00_오이도차고지")
        for path in population if path
    ]

    return best, fitness_with_return, total_distance, total_minutes

    return best, fitness_with_return, total_distance, total_minutes