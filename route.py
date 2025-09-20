import re
import ast

# 거리 데이터 불러오기 (파일 경로는 환경에 맞게 수정)
with open(r"C:\\Users\\panda\\Documents\\졸작\\distance_map_UPDATED.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dict_blocks = re.findall(r"\{[^{}]+\}", raw_text)
distance_matrix = {}
for block in dict_blocks:
    distance_matrix.update(ast.literal_eval(block))


def get_distance_between(stop1, stop2):
    """실제 거리 매트릭스를 기반으로 거리 반환"""
    return distance_matrix.get((stop1, stop2)) 
'''
def get_distance_between(stop1, stop2):
    # 간단한 예시용 거리 계산
    if stop1 == stop2:
        return 0
    return 5  # 고정 거리 '''