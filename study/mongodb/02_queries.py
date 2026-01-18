"""
MongoDB 쿼리 연산 학습

이 파일에서는 MongoDB의 다양한 쿼리 연산자와 조건문을 학습합니다.

쿼리 연산자 종류:
1. 비교 연산자: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
2. 논리 연산자: $and, $or, $not, $nor
3. 요소 연산자: $exists, $type
4. 배열 연산자: $all, $elemMatch, $size
5. 평가 연산자: $regex, $expr, $text
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime, timedelta
from typing import List, Dict, Any


class MongoDBQueries:
    """MongoDB 쿼리 학습 클래스"""

    def __init__(self, uri: str = "mongodb://localhost:27017/"):
        """MongoDB 연결"""
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client["study_db"]
        self.collection = self.db["products"]

        try:
            self.client.admin.command('ping')
            print("[OK] MongoDB 연결 성공")
        except ConnectionFailure as e:
            print(f"[ERROR] 연결 실패: {e}")
            raise

    def setup_sample_data(self):
        """샘플 데이터 설정"""
        print("\n" + "=" * 60)
        print("샘플 데이터 설정")
        print("=" * 60)

        # 기존 데이터 삭제
        self.collection.delete_many({})

        # 샘플 상품 데이터
        products = [
            {
                "name": "노트북",
                "category": "전자제품",
                "price": 1500000,
                "stock": 50,
                "tags": ["컴퓨터", "업무용", "게임"],
                "specs": {"cpu": "i7", "ram": 16, "ssd": 512},
                "ratings": [4.5, 4.8, 4.2, 4.9],
                "is_available": True,
                "created_at": datetime.now() - timedelta(days=30)
            },
            {
                "name": "스마트폰",
                "category": "전자제품",
                "price": 1200000,
                "stock": 100,
                "tags": ["통신", "카메라", "게임"],
                "specs": {"screen": 6.5, "battery": 5000, "storage": 256},
                "ratings": [4.7, 4.6, 4.8],
                "is_available": True,
                "created_at": datetime.now() - timedelta(days=15)
            },
            {
                "name": "무선 이어폰",
                "category": "전자제품",
                "price": 250000,
                "stock": 200,
                "tags": ["음악", "운동"],
                "specs": {"battery_life": 8, "noise_canceling": True},
                "ratings": [4.3, 4.1, 4.5, 4.4, 4.2],
                "is_available": True,
                "created_at": datetime.now() - timedelta(days=7)
            },
            {
                "name": "운동화",
                "category": "의류",
                "price": 150000,
                "stock": 30,
                "tags": ["운동", "러닝"],
                "specs": {"size_range": [250, 280], "color": "black"},
                "ratings": [4.0, 3.8, 4.2],
                "is_available": True,
                "created_at": datetime.now() - timedelta(days=60)
            },
            {
                "name": "겨울 패딩",
                "category": "의류",
                "price": 350000,
                "stock": 0,
                "tags": ["겨울", "아우터"],
                "specs": {"size": ["S", "M", "L", "XL"], "fill_power": 800},
                "ratings": [4.6, 4.7],
                "is_available": False,
                "created_at": datetime.now() - timedelta(days=90)
            },
            {
                "name": "프로그래밍 책",
                "category": "도서",
                "price": 35000,
                "stock": 500,
                "tags": ["IT", "개발", "파이썬"],
                "specs": {"pages": 450, "author": "김개발"},
                "ratings": [4.9, 5.0, 4.8, 4.9],
                "is_available": True,
                "created_at": datetime.now() - timedelta(days=3)
            }
        ]

        result = self.collection.insert_many(products)
        print(f"삽입된 문서 수: {len(result.inserted_ids)}")

    # =========================================================================
    # 비교 연산자
    # =========================================================================

    def comparison_operators(self):
        """
        비교 연산자 예제

        $eq: 같음 (equal)
        $ne: 같지 않음 (not equal)
        $gt: 초과 (greater than)
        $gte: 이상 (greater than or equal)
        $lt: 미만 (less than)
        $lte: 이하 (less than or equal)
        $in: 배열 내 값 중 하나와 일치
        $nin: 배열 내 어떤 값과도 일치하지 않음
        """
        print("\n" + "=" * 60)
        print("1. 비교 연산자")
        print("=" * 60)

        # $eq: 같음 (명시적으로 사용하지 않아도 됨)
        print("\n$eq - 카테고리가 '전자제품'인 상품:")
        for p in self.collection.find({"category": {"$eq": "전자제품"}}):
            print(f"  - {p['name']}")

        # $ne: 같지 않음
        print("\n$ne - 카테고리가 '전자제품'이 아닌 상품:")
        for p in self.collection.find({"category": {"$ne": "전자제품"}}):
            print(f"  - {p['name']} ({p['category']})")

        # $gt, $lt: 초과, 미만
        print("\n$gt, $lt - 가격이 10만원 초과 100만원 미만:")
        query = {"price": {"$gt": 100000, "$lt": 1000000}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원")

        # $gte, $lte: 이상, 이하
        print("\n$gte, $lte - 재고가 50개 이상 200개 이하:")
        query = {"stock": {"$gte": 50, "$lte": 200}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['stock']}개")

        # $in: 배열 내 값 중 하나와 일치
        print("\n$in - 카테고리가 '전자제품' 또는 '도서':")
        query = {"category": {"$in": ["전자제품", "도서"]}}
        for p in self.collection.find(query):
            print(f"  - {p['name']} ({p['category']})")

        # $nin: 배열 내 어떤 값과도 일치하지 않음
        print("\n$nin - 카테고리가 '전자제품', '도서'가 아닌 상품:")
        query = {"category": {"$nin": ["전자제품", "도서"]}}
        for p in self.collection.find(query):
            print(f"  - {p['name']} ({p['category']})")

    # =========================================================================
    # 논리 연산자
    # =========================================================================

    def logical_operators(self):
        """
        논리 연산자 예제

        $and: 모든 조건 만족 (AND)
        $or: 하나 이상의 조건 만족 (OR)
        $not: 조건 부정 (NOT)
        $nor: 모든 조건 불만족 (NOR)
        """
        print("\n" + "=" * 60)
        print("2. 논리 연산자")
        print("=" * 60)

        # $and: 모든 조건 만족
        print("\n$and - 전자제품이면서 가격 50만원 이하:")
        query = {
            "$and": [
                {"category": "전자제품"},
                {"price": {"$lte": 500000}}
            ]
        }
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원")

        # 암시적 $and (같은 필드가 아니면 생략 가능)
        print("\n암시적 $and - 동일한 결과:")
        query = {"category": "전자제품", "price": {"$lte": 500000}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원")

        # $or: 하나 이상의 조건 만족
        print("\n$or - 가격 20만원 이하 또는 재고 없음:")
        query = {
            "$or": [
                {"price": {"$lte": 200000}},
                {"stock": 0}
            ]
        }
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원, 재고: {p['stock']}")

        # $not: 조건 부정
        print("\n$not - 가격이 50만원 이하가 아닌 상품:")
        query = {"price": {"$not": {"$lte": 500000}}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원")

        # $nor: 모든 조건 불만족
        print("\n$nor - 전자제품도 아니고 의류도 아닌 상품:")
        query = {
            "$nor": [
                {"category": "전자제품"},
                {"category": "의류"}
            ]
        }
        for p in self.collection.find(query):
            print(f"  - {p['name']} ({p['category']})")

    # =========================================================================
    # 요소 연산자
    # =========================================================================

    def element_operators(self):
        """
        요소 연산자 예제

        $exists: 필드 존재 여부
        $type: 필드의 BSON 타입 확인
        """
        print("\n" + "=" * 60)
        print("3. 요소 연산자")
        print("=" * 60)

        # $exists: 필드 존재 여부
        print("\n$exists - 'specs.cpu' 필드가 있는 상품:")
        query = {"specs.cpu": {"$exists": True}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: CPU {p['specs']['cpu']}")

        print("\n$exists - 'discount' 필드가 없는 상품:")
        query = {"discount": {"$exists": False}}
        count = self.collection.count_documents(query)
        print(f"  - {count}개 상품")

        # $type: BSON 타입 확인
        # 타입: "string", "int", "double", "bool", "array", "object", "date" 등
        print("\n$type - 'price' 필드가 숫자인 상품:")
        query = {"price": {"$type": "int"}}
        count = self.collection.count_documents(query)
        print(f"  - {count}개 상품")

    # =========================================================================
    # 배열 연산자
    # =========================================================================

    def array_operators(self):
        """
        배열 연산자 예제

        $all: 배열에 지정된 모든 요소 포함
        $elemMatch: 배열 요소가 모든 조건 만족
        $size: 배열 크기
        """
        print("\n" + "=" * 60)
        print("4. 배열 연산자")
        print("=" * 60)

        # $all: 배열에 지정된 모든 요소 포함
        print("\n$all - '운동', '게임' 태그를 모두 가진 상품:")
        query = {"tags": {"$all": ["운동", "게임"]}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['tags']}")

        # 배열 내 값 검색 (단일 값)
        print("\n배열 내 값 검색 - '파이썬' 태그를 가진 상품:")
        query = {"tags": "파이썬"}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['tags']}")

        # $size: 배열 크기
        print("\n$size - 태그가 정확히 3개인 상품:")
        query = {"tags": {"$size": 3}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['tags']}")

        # $elemMatch: 배열 요소가 모든 조건 만족 (복합 조건)
        print("\n$elemMatch - 평점 4.8 이상인 리뷰가 있는 상품:")
        query = {"ratings": {"$elemMatch": {"$gte": 4.8}}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['ratings']}")

    # =========================================================================
    # 평가 연산자
    # =========================================================================

    def evaluation_operators(self):
        """
        평가 연산자 예제

        $regex: 정규식 매칭
        $expr: 집계 표현식 사용
        $mod: 나머지 연산
        """
        print("\n" + "=" * 60)
        print("5. 평가 연산자")
        print("=" * 60)

        # $regex: 정규식 매칭
        print("\n$regex - 이름에 '폰' 또는 '북'이 포함된 상품:")
        query = {"name": {"$regex": "(폰|북)", "$options": "i"}}  # i: 대소문자 무시
        for p in self.collection.find(query):
            print(f"  - {p['name']}")

        # $regex: 시작/끝 패턴
        print("\n$regex - 이름이 '무선'으로 시작하는 상품:")
        query = {"name": {"$regex": "^무선"}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}")

        # $expr: 필드 간 비교
        print("\n$expr - 재고가 100개 이상이고 가격이 30만원 이하인 상품:")
        query = {
            "$expr": {
                "$and": [
                    {"$gte": ["$stock", 100]},
                    {"$lte": ["$price", 300000]}
                ]
            }
        }
        for p in self.collection.find(query):
            print(f"  - {p['name']}: 재고 {p['stock']}, 가격 {p['price']:,}원")

        # $mod: 나머지 연산
        print("\n$mod - 가격이 10만원으로 나누어 떨어지는 상품:")
        query = {"price": {"$mod": [100000, 0]}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: {p['price']:,}원")

    # =========================================================================
    # 중첩 문서/배열 쿼리
    # =========================================================================

    def nested_queries(self):
        """
        중첩 문서 및 배열 쿼리 예제

        점 표기법(dot notation)을 사용하여 중첩된 필드에 접근
        """
        print("\n" + "=" * 60)
        print("6. 중첩 문서/배열 쿼리")
        print("=" * 60)

        # 중첩 문서 필드 접근 (점 표기법)
        print("\n점 표기법 - RAM이 16GB 이상인 상품:")
        query = {"specs.ram": {"$gte": 16}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: RAM {p['specs'].get('ram')}GB")

        # 중첩 문서 필드가 특정 값인 경우
        print("\n점 표기법 - 노이즈 캔슬링 지원 상품:")
        query = {"specs.noise_canceling": True}
        for p in self.collection.find(query):
            print(f"  - {p['name']}")

        # 배열의 특정 인덱스 접근
        print("\n배열 인덱스 - 첫 번째 평점이 4.5 이상인 상품:")
        query = {"ratings.0": {"$gte": 4.5}}
        for p in self.collection.find(query):
            print(f"  - {p['name']}: 첫 평점 {p['ratings'][0]}")

    # =========================================================================
    # 정렬, 제한, 건너뛰기
    # =========================================================================

    def sorting_and_pagination(self):
        """
        정렬 및 페이지네이션 예제
        """
        print("\n" + "=" * 60)
        print("7. 정렬 및 페이지네이션")
        print("=" * 60)

        # 단일 필드 정렬
        print("\n가격 오름차순:")
        for p in self.collection.find().sort("price", 1):  # 1: 오름차순, -1: 내림차순
            print(f"  - {p['name']}: {p['price']:,}원")

        # 다중 필드 정렬
        print("\n카테고리 오름차순, 가격 내림차순:")
        for p in self.collection.find().sort([("category", 1), ("price", -1)]):
            print(f"  - {p['category']}/{p['name']}: {p['price']:,}원")

        # 페이지네이션 (skip + limit)
        page_size = 2
        page_num = 1  # 1부터 시작

        print(f"\n페이지 {page_num} (페이지당 {page_size}개):")
        skip_count = (page_num - 1) * page_size
        for p in self.collection.find().skip(skip_count).limit(page_size):
            print(f"  - {p['name']}")

        print(f"\n페이지 2 (페이지당 {page_size}개):")
        skip_count = (2 - 1) * page_size
        for p in self.collection.find().skip(skip_count).limit(page_size):
            print(f"  - {p['name']}")

    # =========================================================================
    # Projection (필드 선택)
    # =========================================================================

    def projection_examples(self):
        """
        Projection 예제 - 반환할 필드 선택
        """
        print("\n" + "=" * 60)
        print("8. Projection (필드 선택)")
        print("=" * 60)

        # 포함할 필드 지정 (1)
        print("\n이름과 가격만 조회:")
        for p in self.collection.find({}, {"name": 1, "price": 1, "_id": 0}):
            print(f"  - {p}")

        # 제외할 필드 지정 (0)
        print("\nspecs, ratings 제외하고 조회:")
        for p in self.collection.find({}, {"specs": 0, "ratings": 0}).limit(2):
            print(f"  - {p['name']}: {list(p.keys())}")

        # 배열 슬라이싱 ($slice)
        print("\n평점 상위 2개만 조회 ($slice):")
        for p in self.collection.find({}, {"name": 1, "ratings": {"$slice": 2}, "_id": 0}):
            print(f"  - {p['name']}: {p.get('ratings', [])}")

    def cleanup(self):
        """테스트 데이터 정리"""
        self.collection.delete_many({})
        print("\n[INFO] 테스트 데이터 정리 완료")

    def close(self):
        """연결 종료"""
        self.client.close()
        print("[INFO] MongoDB 연결 종료")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MongoDB 쿼리 연산 학습")
    print("=" * 60)

    queries = None
    try:
        queries = MongoDBQueries()

        # 샘플 데이터 설정
        queries.setup_sample_data()

        # 쿼리 예제 실행
        queries.comparison_operators()
        queries.logical_operators()
        queries.element_operators()
        queries.array_operators()
        queries.evaluation_operators()
        queries.nested_queries()
        queries.sorting_and_pagination()
        queries.projection_examples()

        # 정리
        queries.cleanup()

    except ConnectionFailure:
        print("\n[TIP] MongoDB 서버가 실행 중인지 확인하세요.")
    finally:
        if queries:
            queries.close()


if __name__ == "__main__":
    main()
