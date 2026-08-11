"""수동 녹화 타임라인을 의미 단위 workflow step 으로 묶는 오프라인 패키지.

recording_filter 산출(interaction_timeline.json)만 읽고 VLM 을 부르지 않는다 -
그룹핑 규칙은 튜닝 회차가 가장 많은 단계라 재실행이 공짜여야 한다.
"""
