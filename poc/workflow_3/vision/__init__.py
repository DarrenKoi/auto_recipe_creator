"""CV align-key 엔진 — 매칭/ensemble/자산 해석/보정/라이브 탐색.

설계 규칙(2026-05-25 확정): OpenCV 가 정량 점수와 최종 좌표를 결정하고, VLM 은
영역 식별·모호한 FOV 설명·feasibility 평가만 한다. VLM 답변이 낮은 CV 점수를
뒤집거나 반복 가능한 stage 전환을 결정하게 하지 않는다.
"""
