"""
MongoDB 인덱싱 학습

인덱스는 쿼리 성능을 크게 향상시키는 데이터 구조입니다.
적절한 인덱스 없이는 MongoDB가 전체 컬렉션을 스캔해야 합니다.

인덱스 종류:
1. 단일 필드 인덱스 (Single Field Index)
2. 복합 인덱스 (Compound Index)
3. 멀티키 인덱스 (Multikey Index) - 배열 필드
4. 텍스트 인덱스 (Text Index)
5. 지리공간 인덱스 (Geospatial Index)
6. 해시 인덱스 (Hashed Index)
"""

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure
from datetime import datetime, timedelta
import random
import time


class MongoDBIndexing:
    """MongoDB 인덱싱 학습 클래스"""

    def __init__(self, uri: str = "mongodb://localhost:27017/"):
        """MongoDB 연결"""
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client["study_db"]
        self.collection = self.db["index_test"]

        try:
            self.client.admin.command('ping')
            print("[OK] MongoDB 연결 성공")
        except ConnectionFailure as e:
            print(f"[ERROR] 연결 실패: {e}")
            raise

    def setup_sample_data(self, count: int = 10000):
        """대량의 샘플 데이터 생성"""
        print("\n" + "=" * 60)
        print(f"샘플 데이터 생성 ({count}건)")
        print("=" * 60)

        # 기존 데이터 및 인덱스 삭제
        self.collection.drop()

        # 샘플 데이터 생성
        categories = ["전자제품", "의류", "식품", "가구", "도서"]
        statuses = ["active", "inactive", "pending"]
        cities = ["서울", "부산", "대구", "인천", "광주", "대전"]

        documents = []
        for i in range(count):
            doc = {
                "product_id": f"PROD-{i:05d}",
                "name": f"상품 {i}",
                "description": f"이것은 상품 {i}의 상세 설명입니다. 품질이 우수하고 가격이 합리적입니다.",
                "category": random.choice(categories),
                "price": random.randint(10000, 1000000),
                "stock": random.randint(0, 1000),
                "status": random.choice(statuses),
                "tags": random.sample(["인기", "신상", "할인", "추천", "한정"], k=random.randint(1, 3)),
                "seller": {
                    "name": f"판매자{random.randint(1, 100)}",
                    "city": random.choice(cities)
                },
                "created_at": datetime.now() - timedelta(days=random.randint(1, 365)),
                "views": random.randint(0, 10000)
            }
            documents.append(doc)

        # 배치 삽입
        start_time = time.time()
        self.collection.insert_many(documents)
        elapsed = time.time() - start_time

        print(f"데이터 삽입 완료: {count}건, {elapsed:.2f}초")

    # =========================================================================
    # 인덱스 기본
    # =========================================================================

    def list_indexes(self):
        """현재 인덱스 목록 확인"""
        print("\n현재 인덱스 목록:")
        for idx in self.collection.list_indexes():
            print(f"  - {idx['name']}: {idx['key']}")

    def explain_query(self, query: dict, hint: str = None):
        """
        쿼리 실행 계획 분석

        explain()을 사용하여 쿼리가 어떻게 실행되는지 확인
        - COLLSCAN: 전체 컬렉션 스캔 (느림)
        - IXSCAN: 인덱스 스캔 (빠름)
        """
        cursor = self.collection.find(query)
        if hint:
            cursor = cursor.hint(hint)

        explain = cursor.explain()

        winning_plan = explain.get('queryPlanner', {}).get('winningPlan', {})

        # 스테이지 추출
        def get_stage(plan):
            stage = plan.get('stage', 'UNKNOWN')
            if 'inputStage' in plan:
                return f"{stage} -> {get_stage(plan['inputStage'])}"
            return stage

        stage_info = get_stage(winning_plan)

        # 실행 통계 (executionStats가 있는 경우)
        exec_stats = explain.get('executionStats', {})
        docs_examined = exec_stats.get('totalDocsExamined', 'N/A')
        keys_examined = exec_stats.get('totalKeysExamined', 'N/A')
        exec_time = exec_stats.get('executionTimeMillis', 'N/A')

        return {
            'stage': stage_info,
            'docs_examined': docs_examined,
            'keys_examined': keys_examined,
            'exec_time_ms': exec_time
        }

    # =========================================================================
    # 단일 필드 인덱스
    # =========================================================================

    def single_field_index(self):
        """
        단일 필드 인덱스 생성 및 테스트

        - 가장 기본적인 인덱스
        - 하나의 필드에 대해 생성
        - 오름차순(1) 또는 내림차순(-1)
        """
        print("\n" + "=" * 60)
        print("1. 단일 필드 인덱스")
        print("=" * 60)

        query = {"category": "전자제품"}

        # 인덱스 없이 쿼리
        print("\n[인덱스 없음]")
        start = time.time()
        count = self.collection.count_documents(query)
        elapsed = time.time() - start
        explain = self.explain_query(query)
        print(f"  결과: {count}건, 시간: {elapsed*1000:.2f}ms")
        print(f"  실행 계획: {explain['stage']}")
        print(f"  검사한 문서: {explain['docs_examined']}")

        # 인덱스 생성
        print("\n인덱스 생성: category_1")
        self.collection.create_index([("category", ASCENDING)], name="category_1")

        # 인덱스로 쿼리
        print("\n[인덱스 사용]")
        start = time.time()
        count = self.collection.count_documents(query)
        elapsed = time.time() - start
        explain = self.explain_query(query)
        print(f"  결과: {count}건, 시간: {elapsed*1000:.2f}ms")
        print(f"  실행 계획: {explain['stage']}")
        print(f"  검사한 키: {explain['keys_examined']}")

    # =========================================================================
    # 복합 인덱스
    # =========================================================================

    def compound_index(self):
        """
        복합 인덱스 (Compound Index)

        - 여러 필드를 조합한 인덱스
        - 필드 순서가 중요 (왼쪽 접두사 규칙)
        - {a: 1, b: 1, c: 1} 인덱스는 {a}, {a, b}, {a, b, c} 쿼리 지원
        """
        print("\n" + "=" * 60)
        print("2. 복합 인덱스")
        print("=" * 60)

        # 복합 쿼리
        query = {"category": "전자제품", "status": "active"}

        print("\n[단일 인덱스만 있을 때]")
        explain = self.explain_query(query)
        print(f"  실행 계획: {explain['stage']}")

        # 복합 인덱스 생성
        print("\n인덱스 생성: category_status")
        self.collection.create_index(
            [("category", ASCENDING), ("status", ASCENDING)],
            name="category_status"
        )

        print("\n[복합 인덱스 사용]")
        start = time.time()
        count = self.collection.count_documents(query)
        elapsed = time.time() - start
        explain = self.explain_query(query)
        print(f"  결과: {count}건, 시간: {elapsed*1000:.2f}ms")
        print(f"  실행 계획: {explain['stage']}")

        # 정렬을 포함한 복합 인덱스
        print("\n정렬 + 필터 쿼리:")
        print("인덱스 생성: category_price (카테고리 오름차순, 가격 내림차순)")
        self.collection.create_index(
            [("category", ASCENDING), ("price", DESCENDING)],
            name="category_price"
        )

        # 인덱스를 활용한 정렬
        cursor = self.collection.find(
            {"category": "전자제품"}
        ).sort("price", -1).limit(5)

        print("\n전자제품 가격 내림차순 상위 5개:")
        for doc in cursor:
            print(f"  {doc['name']}: {doc['price']:,}원")

    # =========================================================================
    # 유니크 인덱스
    # =========================================================================

    def unique_index(self):
        """
        유니크 인덱스 (Unique Index)

        - 중복 값을 허용하지 않음
        - 데이터 무결성 보장
        """
        print("\n" + "=" * 60)
        print("3. 유니크 인덱스")
        print("=" * 60)

        # 유니크 인덱스 생성
        print("\n유니크 인덱스 생성: product_id")
        self.collection.create_index(
            [("product_id", ASCENDING)],
            unique=True,
            name="product_id_unique"
        )

        # 중복 삽입 시도
        print("\n중복 데이터 삽입 시도:")
        try:
            self.collection.insert_one({"product_id": "PROD-00001", "name": "중복 테스트"})
            print("  삽입 성공 (예상치 못함)")
        except Exception as e:
            print(f"  예상된 오류: {type(e).__name__}")
            print(f"  메시지: 중복 키 오류 (product_id: PROD-00001)")

    # =========================================================================
    # 텍스트 인덱스
    # =========================================================================

    def text_index(self):
        """
        텍스트 인덱스 (Text Index)

        - 텍스트 검색을 위한 인덱스
        - 컬렉션당 하나의 텍스트 인덱스만 가능
        - 여러 필드를 포함할 수 있음
        """
        print("\n" + "=" * 60)
        print("4. 텍스트 인덱스")
        print("=" * 60)

        # 텍스트 인덱스 생성
        print("\n텍스트 인덱스 생성: name + description")
        self.collection.create_index(
            [("name", TEXT), ("description", TEXT)],
            name="text_search",
            default_language="korean"  # 한국어 지원
        )

        # 텍스트 검색
        print("\n텍스트 검색: '품질'")
        results = self.collection.find(
            {"$text": {"$search": "품질"}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(5)

        for doc in results:
            print(f"  {doc['name']}: {doc.get('score', 0):.2f}")

        # 여러 단어 검색
        print("\n텍스트 검색: '품질 가격' (OR 검색)")
        results = self.collection.find(
            {"$text": {"$search": "품질 가격"}}
        ).limit(3)

        for doc in results:
            print(f"  {doc['name']}")

    # =========================================================================
    # 멀티키 인덱스
    # =========================================================================

    def multikey_index(self):
        """
        멀티키 인덱스 (Multikey Index)

        - 배열 필드에 대한 인덱스
        - 배열의 각 요소에 대해 인덱스 항목 생성
        - 자동으로 멀티키 인덱스로 인식
        """
        print("\n" + "=" * 60)
        print("5. 멀티키 인덱스 (배열)")
        print("=" * 60)

        # tags 필드 (배열)에 인덱스 생성
        print("\n멀티키 인덱스 생성: tags")
        self.collection.create_index([("tags", ASCENDING)], name="tags_1")

        # 배열 요소 검색
        print("\n'인기' 태그를 가진 상품 검색:")
        start = time.time()
        results = list(self.collection.find({"tags": "인기"}).limit(5))
        elapsed = time.time() - start

        print(f"  결과: {len(results)}건 (상위 5개), 시간: {elapsed*1000:.2f}ms")
        for doc in results:
            print(f"    {doc['name']}: {doc['tags']}")

        # $all 연산자와 함께 사용
        print("\n'인기' AND '할인' 태그를 모두 가진 상품:")
        results = list(self.collection.find(
            {"tags": {"$all": ["인기", "할인"]}}
        ).limit(5))

        print(f"  결과: {len(results)}건 (상위 5개)")
        for doc in results:
            print(f"    {doc['name']}: {doc['tags']}")

    # =========================================================================
    # 부분 인덱스 (Partial Index)
    # =========================================================================

    def partial_index(self):
        """
        부분 인덱스 (Partial Index)

        - 특정 조건을 만족하는 문서에만 인덱스 생성
        - 인덱스 크기 감소, 성능 향상
        """
        print("\n" + "=" * 60)
        print("6. 부분 인덱스 (Partial Index)")
        print("=" * 60)

        # 활성 상태인 문서에만 인덱스 생성
        print("\n부분 인덱스 생성: active 상태인 문서의 views 필드만")
        self.collection.create_index(
            [("views", DESCENDING)],
            name="views_active_only",
            partialFilterExpression={"status": "active"}
        )

        # 부분 인덱스 활용 쿼리
        print("\n활성 상태 상품 중 조회수 상위 5개:")
        results = self.collection.find(
            {"status": "active"}
        ).sort("views", -1).limit(5)

        for doc in results:
            print(f"  {doc['name']}: {doc['views']:,} views")

    # =========================================================================
    # 중첩 문서 인덱스
    # =========================================================================

    def nested_document_index(self):
        """
        중첩 문서 필드에 대한 인덱스

        - 점 표기법(dot notation)으로 중첩 필드 지정
        """
        print("\n" + "=" * 60)
        print("7. 중첩 문서 인덱스")
        print("=" * 60)

        # 중첩 필드 인덱스 생성
        print("\n인덱스 생성: seller.city")
        self.collection.create_index(
            [("seller.city", ASCENDING)],
            name="seller_city"
        )

        # 중첩 필드 쿼리
        print("\n서울 판매자의 상품:")
        start = time.time()
        count = self.collection.count_documents({"seller.city": "서울"})
        elapsed = time.time() - start

        print(f"  결과: {count}건, 시간: {elapsed*1000:.2f}ms")

    # =========================================================================
    # TTL 인덱스
    # =========================================================================

    def ttl_index(self):
        """
        TTL 인덱스 (Time-To-Live)

        - 일정 시간이 지난 문서를 자동으로 삭제
        - 로그, 세션, 임시 데이터에 유용
        """
        print("\n" + "=" * 60)
        print("8. TTL 인덱스")
        print("=" * 60)

        # 임시 컬렉션 생성
        temp_collection = self.db["sessions"]
        temp_collection.drop()

        # TTL 인덱스 생성 (60초 후 만료)
        print("\nTTL 인덱스 생성: 60초 후 자동 삭제")
        temp_collection.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=60,
            name="session_ttl"
        )

        # 세션 데이터 삽입
        temp_collection.insert_one({
            "session_id": "abc123",
            "user_id": 1,
            "created_at": datetime.now()
        })

        print("  세션 데이터 삽입 완료")
        print("  60초 후 자동 삭제됩니다 (백그라운드 프로세스)")

        # 인덱스 확인
        print("\nTTL 인덱스 확인:")
        for idx in temp_collection.list_indexes():
            if "expireAfterSeconds" in idx:
                print(f"  {idx['name']}: {idx['expireAfterSeconds']}초 후 만료")

    # =========================================================================
    # 인덱스 관리
    # =========================================================================

    def index_management(self):
        """
        인덱스 관리 - 조회, 삭제, 재구축
        """
        print("\n" + "=" * 60)
        print("9. 인덱스 관리")
        print("=" * 60)

        # 인덱스 목록 조회
        print("\n현재 인덱스 목록:")
        for idx in self.collection.list_indexes():
            size_info = ""
            print(f"  - {idx['name']}: {dict(idx['key'])} {size_info}")

        # 인덱스 통계
        print("\n인덱스 통계:")
        stats = self.db.command("collStats", "index_test")
        total_index_size = stats.get("totalIndexSize", 0)
        print(f"  총 인덱스 크기: {total_index_size / 1024:.2f} KB")

        # 개별 인덱스 삭제
        print("\n인덱스 삭제: category_1")
        try:
            self.collection.drop_index("category_1")
            print("  삭제 완료")
        except Exception as e:
            print(f"  오류: {e}")

        # 모든 인덱스 삭제 (_id 제외)
        # self.collection.drop_indexes()

    def best_practices(self):
        """
        인덱스 모범 사례
        """
        print("\n" + "=" * 60)
        print("10. 인덱스 모범 사례")
        print("=" * 60)

        tips = """
1. ESR 규칙 (Equality, Sort, Range)
   - 복합 인덱스 필드 순서: 동등 조건 → 정렬 → 범위 조건
   - 예: {status: 1, created_at: -1, price: 1}

2. 선택성(Selectivity)이 높은 필드 우선
   - 고유한 값이 많은 필드를 앞에 배치
   - status(3개 값)보다 user_id(많은 값)가 선택성이 높음

3. 커버링 인덱스 활용
   - 쿼리에 필요한 모든 필드를 인덱스에 포함
   - 문서를 읽지 않고 인덱스만으로 결과 반환

4. 인덱스 개수 제한
   - 쓰기 성능에 영향 (각 인덱스 업데이트 필요)
   - 일반적으로 컬렉션당 5-10개 이하 권장

5. 불필요한 인덱스 제거
   - 사용하지 않는 인덱스는 오버헤드만 발생
   - $indexStats로 사용 빈도 확인

6. 백그라운드 인덱스 생성
   - 대용량 컬렉션에서는 background=True 사용
   - 프로덕션 환경에서 락 방지
        """
        print(tips)

    def cleanup(self):
        """테스트 데이터 정리"""
        self.collection.drop()
        self.db["sessions"].drop()
        print("\n[INFO] 테스트 데이터 정리 완료")

    def close(self):
        """연결 종료"""
        self.client.close()
        print("[INFO] MongoDB 연결 종료")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MongoDB 인덱싱 학습")
    print("=" * 60)

    idx = None
    try:
        idx = MongoDBIndexing()

        # 샘플 데이터 설정
        idx.setup_sample_data(count=10000)

        # 인덱스 학습
        idx.single_field_index()
        idx.compound_index()
        idx.unique_index()
        idx.text_index()
        idx.multikey_index()
        idx.partial_index()
        idx.nested_document_index()
        idx.ttl_index()

        # 인덱스 관리
        idx.index_management()
        idx.best_practices()

        # 정리
        idx.cleanup()

    except ConnectionFailure:
        print("\n[TIP] MongoDB 서버가 실행 중인지 확인하세요.")
    finally:
        if idx:
            idx.close()


if __name__ == "__main__":
    main()
