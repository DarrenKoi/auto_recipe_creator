"""
MongoDB 집계 파이프라인(Aggregation Pipeline) 학습

집계 파이프라인은 문서들을 여러 단계(stage)를 거쳐 처리하는 프레임워크입니다.
SQL의 GROUP BY, JOIN, 서브쿼리 등의 기능을 수행할 수 있습니다.

주요 스테이지:
- $match: 필터링 (WHERE)
- $group: 그룹화 (GROUP BY)
- $project: 필드 선택/변환 (SELECT)
- $sort: 정렬 (ORDER BY)
- $limit, $skip: 페이지네이션
- $lookup: 조인 (JOIN)
- $unwind: 배열 펼치기
- $addFields: 필드 추가
- $bucket: 버킷팅
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random


class MongoDBAggregation:
    """MongoDB 집계 파이프라인 학습 클래스"""

    def __init__(self, uri: str = "mongodb://localhost:27017/"):
        """MongoDB 연결"""
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client["study_db"]

        try:
            self.client.admin.command('ping')
            print("[OK] MongoDB 연결 성공")
        except ConnectionFailure as e:
            print(f"[ERROR] 연결 실패: {e}")
            raise

    def setup_sample_data(self):
        """샘플 데이터 설정 - 주문 및 상품 데이터"""
        print("\n" + "=" * 60)
        print("샘플 데이터 설정")
        print("=" * 60)

        # 기존 데이터 삭제
        self.db["orders"].delete_many({})
        self.db["products"].delete_many({})
        self.db["customers"].delete_many({})

        # 고객 데이터
        customers = [
            {"_id": 1, "name": "김철수", "city": "서울", "tier": "gold"},
            {"_id": 2, "name": "이영희", "city": "부산", "tier": "silver"},
            {"_id": 3, "name": "박민수", "city": "서울", "tier": "gold"},
            {"_id": 4, "name": "정수진", "city": "대전", "tier": "bronze"},
            {"_id": 5, "name": "최동현", "city": "서울", "tier": "silver"},
        ]
        self.db["customers"].insert_many(customers)

        # 상품 데이터
        products = [
            {"_id": 101, "name": "노트북", "category": "전자제품", "price": 1500000},
            {"_id": 102, "name": "마우스", "category": "전자제품", "price": 50000},
            {"_id": 103, "name": "키보드", "category": "전자제품", "price": 100000},
            {"_id": 104, "name": "모니터", "category": "전자제품", "price": 400000},
            {"_id": 105, "name": "의자", "category": "가구", "price": 300000},
            {"_id": 106, "name": "책상", "category": "가구", "price": 500000},
        ]
        self.db["products"].insert_many(products)

        # 주문 데이터
        orders = []
        statuses = ["completed", "pending", "cancelled"]
        for i in range(1, 21):
            order = {
                "_id": i,
                "customer_id": random.choice([1, 2, 3, 4, 5]),
                "items": [
                    {
                        "product_id": random.choice([101, 102, 103, 104, 105, 106]),
                        "quantity": random.randint(1, 3),
                        "price": random.choice([50000, 100000, 300000, 400000, 500000, 1500000])
                    }
                    for _ in range(random.randint(1, 3))
                ],
                "status": random.choice(statuses),
                "order_date": datetime.now() - timedelta(days=random.randint(1, 90))
            }
            # 총액 계산
            order["total"] = sum(item["price"] * item["quantity"] for item in order["items"])
            orders.append(order)

        self.db["orders"].insert_many(orders)
        print(f"고객: {len(customers)}명, 상품: {len(products)}개, 주문: {len(orders)}건")

    # =========================================================================
    # 기본 집계 스테이지
    # =========================================================================

    def match_stage(self):
        """
        $match 스테이지 - 필터링

        SQL의 WHERE 절과 유사
        파이프라인 초반에 사용하면 성능 향상
        """
        print("\n" + "=" * 60)
        print("1. $match - 필터링")
        print("=" * 60)

        # 완료된 주문만 필터링
        pipeline = [
            {"$match": {"status": "completed"}}
        ]

        results = list(self.db["orders"].aggregate(pipeline))
        print(f"\n완료된 주문 수: {len(results)}건")

        # 복합 조건
        pipeline = [
            {
                "$match": {
                    "status": "completed",
                    "total": {"$gte": 500000}
                }
            }
        ]

        results = list(self.db["orders"].aggregate(pipeline))
        print(f"완료된 주문 중 50만원 이상: {len(results)}건")

    def group_stage(self):
        """
        $group 스테이지 - 그룹화

        SQL의 GROUP BY와 유사
        집계 함수: $sum, $avg, $min, $max, $count, $push, $addToSet
        """
        print("\n" + "=" * 60)
        print("2. $group - 그룹화")
        print("=" * 60)

        # 상태별 주문 수와 총액
        pipeline = [
            {
                "$group": {
                    "_id": "$status",  # 그룹 기준
                    "count": {"$sum": 1},  # 개수
                    "total_amount": {"$sum": "$total"},  # 합계
                    "avg_amount": {"$avg": "$total"}  # 평균
                }
            }
        ]

        print("\n상태별 주문 통계:")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  {doc['_id']}: {doc['count']}건, "
                  f"총액: {doc['total_amount']:,}원, "
                  f"평균: {doc['avg_amount']:,.0f}원")

        # 고객별 주문 통계
        pipeline = [
            {"$match": {"status": "completed"}},
            {
                "$group": {
                    "_id": "$customer_id",
                    "order_count": {"$sum": 1},
                    "total_spent": {"$sum": "$total"},
                    "orders": {"$push": "$_id"}  # 주문 ID 목록
                }
            },
            {"$sort": {"total_spent": -1}}  # 총액 내림차순
        ]

        print("\n고객별 완료 주문 통계 (총액 내림차순):")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  고객 {doc['_id']}: {doc['order_count']}건, "
                  f"총액: {doc['total_spent']:,}원")

    def project_stage(self):
        """
        $project 스테이지 - 필드 선택/변환

        필드 포함/제외, 새 필드 생성, 연산 수행
        """
        print("\n" + "=" * 60)
        print("3. $project - 필드 선택/변환")
        print("=" * 60)

        pipeline = [
            {
                "$project": {
                    "_id": 0,
                    "order_id": "$_id",  # 필드 이름 변경
                    "customer_id": 1,
                    "status": 1,
                    "total": 1,
                    "item_count": {"$size": "$items"},  # 배열 크기
                    "is_large_order": {"$gte": ["$total", 500000]}  # 조건 계산
                }
            },
            {"$limit": 5}
        ]

        print("\n주문 정보 (변환됨):")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  주문 {doc['order_id']}: {doc['total']:,}원, "
                  f"아이템 {doc['item_count']}개, "
                  f"대량주문: {doc['is_large_order']}")

    def sort_limit_skip(self):
        """
        $sort, $limit, $skip 스테이지 - 정렬 및 페이지네이션
        """
        print("\n" + "=" * 60)
        print("4. $sort, $limit, $skip - 정렬 및 페이지네이션")
        print("=" * 60)

        # 최근 주문 5건
        pipeline = [
            {"$sort": {"order_date": -1}},  # 날짜 내림차순
            {"$limit": 5},
            {
                "$project": {
                    "_id": 1,
                    "customer_id": 1,
                    "total": 1,
                    "order_date": 1
                }
            }
        ]

        print("\n최근 주문 5건:")
        for doc in self.db["orders"].aggregate(pipeline):
            date_str = doc['order_date'].strftime('%Y-%m-%d')
            print(f"  주문 {doc['_id']}: {doc['total']:,}원 ({date_str})")

        # 페이지네이션 (2페이지, 페이지당 3건)
        page = 2
        page_size = 3
        pipeline = [
            {"$sort": {"_id": 1}},
            {"$skip": (page - 1) * page_size},
            {"$limit": page_size},
            {"$project": {"_id": 1, "total": 1}}
        ]

        print(f"\n페이지 {page} (페이지당 {page_size}건):")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  주문 {doc['_id']}: {doc['total']:,}원")

    # =========================================================================
    # 고급 집계 스테이지
    # =========================================================================

    def lookup_stage(self):
        """
        $lookup 스테이지 - 조인

        다른 컬렉션과 조인 (SQL의 LEFT OUTER JOIN과 유사)
        """
        print("\n" + "=" * 60)
        print("5. $lookup - 조인")
        print("=" * 60)

        # 주문에 고객 정보 조인
        pipeline = [
            {"$match": {"status": "completed"}},
            {
                "$lookup": {
                    "from": "customers",  # 조인할 컬렉션
                    "localField": "customer_id",  # 현재 컬렉션의 필드
                    "foreignField": "_id",  # 조인할 컬렉션의 필드
                    "as": "customer_info"  # 결과 필드명
                }
            },
            {"$unwind": "$customer_info"},  # 배열을 문서로 펼침
            {
                "$project": {
                    "_id": 1,
                    "total": 1,
                    "customer_name": "$customer_info.name",
                    "customer_city": "$customer_info.city",
                    "customer_tier": "$customer_info.tier"
                }
            },
            {"$limit": 5}
        ]

        print("\n주문 + 고객 정보:")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  주문 {doc['_id']}: {doc['customer_name']} ({doc['customer_city']}) "
                  f"- {doc['total']:,}원 [{doc['customer_tier']}]")

    def unwind_stage(self):
        """
        $unwind 스테이지 - 배열 펼치기

        배열의 각 요소를 별도의 문서로 분리
        """
        print("\n" + "=" * 60)
        print("6. $unwind - 배열 펼치기")
        print("=" * 60)

        # 주문의 각 아이템을 별도 문서로
        pipeline = [
            {"$match": {"_id": 1}},  # 주문 1번만
            {"$unwind": "$items"},
            {
                "$project": {
                    "order_id": "$_id",
                    "product_id": "$items.product_id",
                    "quantity": "$items.quantity",
                    "price": "$items.price"
                }
            }
        ]

        print("\n주문 1번의 아이템들 (펼침):")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  상품 {doc['product_id']}: {doc['quantity']}개 × "
                  f"{doc['price']:,}원")

        # 전체 아이템별 판매량
        pipeline = [
            {"$match": {"status": "completed"}},
            {"$unwind": "$items"},
            {
                "$group": {
                    "_id": "$items.product_id",
                    "total_quantity": {"$sum": "$items.quantity"},
                    "total_revenue": {
                        "$sum": {"$multiply": ["$items.quantity", "$items.price"]}
                    }
                }
            },
            {"$sort": {"total_revenue": -1}},
            {
                "$lookup": {
                    "from": "products",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "product"
                }
            },
            {"$unwind": "$product"},
            {
                "$project": {
                    "product_name": "$product.name",
                    "total_quantity": 1,
                    "total_revenue": 1
                }
            }
        ]

        print("\n상품별 판매 통계 (완료 주문):")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  {doc['product_name']}: {doc['total_quantity']}개, "
                  f"매출 {doc['total_revenue']:,}원")

    def add_fields_stage(self):
        """
        $addFields 스테이지 - 필드 추가

        기존 필드를 유지하면서 새 필드 추가
        """
        print("\n" + "=" * 60)
        print("7. $addFields - 필드 추가")
        print("=" * 60)

        pipeline = [
            {
                "$addFields": {
                    "item_count": {"$size": "$items"},
                    "avg_item_price": {
                        "$avg": "$items.price"
                    },
                    "order_year": {"$year": "$order_date"},
                    "order_month": {"$month": "$order_date"}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "total": 1,
                    "item_count": 1,
                    "avg_item_price": 1,
                    "order_year": 1,
                    "order_month": 1
                }
            },
            {"$limit": 5}
        ]

        print("\n주문 + 계산된 필드:")
        for doc in self.db["orders"].aggregate(pipeline):
            print(f"  주문 {doc['_id']}: 아이템 {doc['item_count']}개, "
                  f"평균가 {doc.get('avg_item_price', 0):,.0f}원, "
                  f"{doc['order_year']}-{doc['order_month']:02d}")

    def bucket_stage(self):
        """
        $bucket 스테이지 - 버킷팅 (구간 그룹화)

        값을 지정된 구간으로 나누어 그룹화
        """
        print("\n" + "=" * 60)
        print("8. $bucket - 버킷팅")
        print("=" * 60)

        # 주문 금액대별 분류
        pipeline = [
            {
                "$bucket": {
                    "groupBy": "$total",
                    "boundaries": [0, 200000, 500000, 1000000, float('inf')],
                    "default": "기타",
                    "output": {
                        "count": {"$sum": 1},
                        "orders": {"$push": "$_id"},
                        "avg_total": {"$avg": "$total"}
                    }
                }
            }
        ]

        print("\n주문 금액대별 분류:")
        labels = {0: "20만원 미만", 200000: "20-50만원", 500000: "50-100만원", 1000000: "100만원 이상"}
        for doc in self.db["orders"].aggregate(pipeline):
            bucket_id = doc['_id']
            label = labels.get(bucket_id, str(bucket_id))
            print(f"  {label}: {doc['count']}건, "
                  f"평균 {doc['avg_total']:,.0f}원")

    def facet_stage(self):
        """
        $facet 스테이지 - 다중 파이프라인 실행

        하나의 입력으로 여러 집계 파이프라인을 병렬 실행
        """
        print("\n" + "=" * 60)
        print("9. $facet - 다중 파이프라인")
        print("=" * 60)

        pipeline = [
            {
                "$facet": {
                    # 파이프라인 1: 상태별 통계
                    "by_status": [
                        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
                    ],
                    # 파이프라인 2: 총 통계
                    "overall": [
                        {
                            "$group": {
                                "_id": None,
                                "total_orders": {"$sum": 1},
                                "total_revenue": {"$sum": "$total"},
                                "avg_order": {"$avg": "$total"}
                            }
                        }
                    ],
                    # 파이프라인 3: 최근 3건
                    "recent": [
                        {"$sort": {"order_date": -1}},
                        {"$limit": 3},
                        {"$project": {"_id": 1, "total": 1}}
                    ]
                }
            }
        ]

        result = list(self.db["orders"].aggregate(pipeline))[0]

        print("\n상태별 통계:")
        for stat in result['by_status']:
            print(f"  {stat['_id']}: {stat['count']}건")

        print("\n전체 통계:")
        overall = result['overall'][0]
        print(f"  총 주문: {overall['total_orders']}건")
        print(f"  총 매출: {overall['total_revenue']:,}원")
        print(f"  평균 주문액: {overall['avg_order']:,.0f}원")

        print("\n최근 주문:")
        for order in result['recent']:
            print(f"  주문 {order['_id']}: {order['total']:,}원")

    # =========================================================================
    # 실전 예제
    # =========================================================================

    def real_world_example(self):
        """
        실전 예제 - 월별 매출 대시보드
        """
        print("\n" + "=" * 60)
        print("10. 실전 예제 - 월별 매출 대시보드")
        print("=" * 60)

        pipeline = [
            # 완료된 주문만
            {"$match": {"status": "completed"}},

            # 날짜 필드 추가
            {
                "$addFields": {
                    "year_month": {
                        "$dateToString": {"format": "%Y-%m", "date": "$order_date"}
                    }
                }
            },

            # 월별 그룹화
            {
                "$group": {
                    "_id": "$year_month",
                    "order_count": {"$sum": 1},
                    "total_revenue": {"$sum": "$total"},
                    "avg_order_value": {"$avg": "$total"},
                    "unique_customers": {"$addToSet": "$customer_id"}
                }
            },

            # 고객 수 계산
            {
                "$addFields": {
                    "customer_count": {"$size": "$unique_customers"}
                }
            },

            # 정렬
            {"$sort": {"_id": -1}},

            # 필드 정리
            {
                "$project": {
                    "_id": 0,
                    "month": "$_id",
                    "order_count": 1,
                    "total_revenue": 1,
                    "avg_order_value": {"$round": ["$avg_order_value", 0]},
                    "customer_count": 1
                }
            }
        ]

        print("\n월별 매출 대시보드:")
        print("-" * 70)
        print(f"{'월':^10} | {'주문수':^8} | {'매출':^15} | {'평균주문':^12} | {'고객수':^6}")
        print("-" * 70)

        for doc in self.db["orders"].aggregate(pipeline):
            print(f"{doc['month']:^10} | {doc['order_count']:^8} | "
                  f"{doc['total_revenue']:>13,}원 | "
                  f"{doc['avg_order_value']:>10,.0f}원 | {doc['customer_count']:^6}")

    def cleanup(self):
        """테스트 데이터 정리"""
        self.db["orders"].delete_many({})
        self.db["products"].delete_many({})
        self.db["customers"].delete_many({})
        print("\n[INFO] 테스트 데이터 정리 완료")

    def close(self):
        """연결 종료"""
        self.client.close()
        print("[INFO] MongoDB 연결 종료")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MongoDB 집계 파이프라인 학습")
    print("=" * 60)

    agg = None
    try:
        agg = MongoDBAggregation()

        # 샘플 데이터 설정
        agg.setup_sample_data()

        # 기본 스테이지
        agg.match_stage()
        agg.group_stage()
        agg.project_stage()
        agg.sort_limit_skip()

        # 고급 스테이지
        agg.lookup_stage()
        agg.unwind_stage()
        agg.add_fields_stage()
        agg.bucket_stage()
        agg.facet_stage()

        # 실전 예제
        agg.real_world_example()

        # 정리
        agg.cleanup()

    except ConnectionFailure:
        print("\n[TIP] MongoDB 서버가 실행 중인지 확인하세요.")
    finally:
        if agg:
            agg.close()


if __name__ == "__main__":
    main()
