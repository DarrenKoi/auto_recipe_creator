"""
MongoDB 기본 CRUD 연산 학습

이 파일에서는 MongoDB의 기본적인 Create, Read, Update, Delete 연산을 학습합니다.

실행 전 필요 사항:
- MongoDB 서버 실행 중
- pymongo 설치: pip install pymongo
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson.objectid import ObjectId
from datetime import datetime
from typing import Optional, Dict, Any, List


class MongoDBCRUD:
    """MongoDB CRUD 연산 학습 클래스"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "study_db"
    ):
        """
        MongoDB 연결 초기화

        Args:
            host: MongoDB 호스트
            port: MongoDB 포트
            username: 사용자명 (선택)
            password: 비밀번호 (선택)
            database: 데이터베이스 이름
        """
        # 연결 문자열 생성
        if username and password:
            uri = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            uri = f"mongodb://{host}:{port}/"

        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[database]

        # 연결 테스트
        try:
            self.client.admin.command('ping')
            print(f"[OK] MongoDB 연결 성공: {host}:{port}/{database}")
        except ConnectionFailure as e:
            print(f"[ERROR] MongoDB 연결 실패: {e}")
            raise

    def close(self):
        """연결 종료"""
        self.client.close()
        print("[INFO] MongoDB 연결 종료")

    # =========================================================================
    # CREATE 연산
    # =========================================================================

    def insert_one_example(self):
        """
        단일 문서 삽입 예제

        insert_one(): 하나의 문서를 컬렉션에 삽입
        - 반환값: InsertOneResult (inserted_id 포함)
        """
        print("\n" + "=" * 60)
        print("1. insert_one() - 단일 문서 삽입")
        print("=" * 60)

        collection = self.db["users"]

        # 삽입할 문서
        user = {
            "name": "홍길동",
            "email": "hong@example.com",
            "age": 30,
            "address": {
                "city": "서울",
                "district": "강남구"
            },
            "hobbies": ["독서", "운동"],
            "created_at": datetime.now()
        }

        # 문서 삽입
        result = collection.insert_one(user)

        print(f"삽입된 문서 ID: {result.inserted_id}")
        print(f"acknowledged: {result.acknowledged}")

        return result.inserted_id

    def insert_many_example(self):
        """
        여러 문서 삽입 예제

        insert_many(): 여러 문서를 한 번에 삽입
        - ordered=True (기본값): 순서대로 삽입, 에러 시 중단
        - ordered=False: 에러가 발생해도 나머지 문서 삽입 계속
        """
        print("\n" + "=" * 60)
        print("2. insert_many() - 여러 문서 삽입")
        print("=" * 60)

        collection = self.db["users"]

        # 삽입할 문서들
        users = [
            {
                "name": "이순신",
                "email": "lee@example.com",
                "age": 45,
                "hobbies": ["독서", "낚시"],
                "created_at": datetime.now()
            },
            {
                "name": "강감찬",
                "email": "kang@example.com",
                "age": 50,
                "hobbies": ["바둑", "등산"],
                "created_at": datetime.now()
            },
            {
                "name": "유관순",
                "email": "yu@example.com",
                "age": 18,
                "hobbies": ["독립운동"],
                "created_at": datetime.now()
            }
        ]

        # 문서들 삽입
        result = collection.insert_many(users)

        print(f"삽입된 문서 수: {len(result.inserted_ids)}")
        print(f"삽입된 IDs: {result.inserted_ids}")

        return result.inserted_ids

    # =========================================================================
    # READ 연산
    # =========================================================================

    def find_one_example(self):
        """
        단일 문서 조회 예제

        find_one(): 조건에 맞는 첫 번째 문서 반환
        - 문서가 없으면 None 반환
        """
        print("\n" + "=" * 60)
        print("3. find_one() - 단일 문서 조회")
        print("=" * 60)

        collection = self.db["users"]

        # 이름으로 조회
        user = collection.find_one({"name": "홍길동"})
        print(f"이름으로 조회: {user}")

        # _id로 조회 (ObjectId 사용)
        if user:
            user_by_id = collection.find_one({"_id": user["_id"]})
            print(f"ID로 조회: {user_by_id}")

        # 특정 필드만 조회 (projection)
        user_projection = collection.find_one(
            {"name": "홍길동"},
            {"name": 1, "email": 1, "_id": 0}  # 1: 포함, 0: 제외
        )
        print(f"특정 필드만 조회: {user_projection}")

        return user

    def find_many_example(self):
        """
        여러 문서 조회 예제

        find(): 조건에 맞는 모든 문서의 커서 반환
        - 커서는 이터레이터처럼 사용
        - limit(), skip(), sort() 등 체이닝 가능
        """
        print("\n" + "=" * 60)
        print("4. find() - 여러 문서 조회")
        print("=" * 60)

        collection = self.db["users"]

        # 모든 문서 조회
        print("\n모든 문서:")
        for user in collection.find():
            print(f"  - {user.get('name')}: {user.get('email')}")

        # 조건으로 조회
        print("\n나이 30 이상:")
        for user in collection.find({"age": {"$gte": 30}}):
            print(f"  - {user.get('name')}: {user.get('age')}세")

        # 정렬 및 제한
        print("\n나이 내림차순, 상위 2명:")
        cursor = collection.find().sort("age", -1).limit(2)
        for user in cursor:
            print(f"  - {user.get('name')}: {user.get('age')}세")

        # 건너뛰기 (페이징)
        print("\n2번째부터 2명 (skip=1, limit=2):")
        cursor = collection.find().skip(1).limit(2)
        for user in cursor:
            print(f"  - {user.get('name')}")

        # 문서 개수
        count = collection.count_documents({"age": {"$gte": 30}})
        print(f"\n30세 이상 인원 수: {count}명")

    # =========================================================================
    # UPDATE 연산
    # =========================================================================

    def update_one_example(self):
        """
        단일 문서 수정 예제

        update_one(): 조건에 맞는 첫 번째 문서 수정
        - $set: 필드 값 설정
        - $inc: 숫자 증가/감소
        - $push: 배열에 요소 추가
        - $pull: 배열에서 요소 제거
        - $unset: 필드 삭제
        """
        print("\n" + "=" * 60)
        print("5. update_one() - 단일 문서 수정")
        print("=" * 60)

        collection = self.db["users"]

        # $set: 필드 값 설정
        result = collection.update_one(
            {"name": "홍길동"},
            {"$set": {"age": 31, "updated_at": datetime.now()}}
        )
        print(f"$set - 매칭: {result.matched_count}, 수정: {result.modified_count}")

        # $inc: 숫자 증가
        result = collection.update_one(
            {"name": "홍길동"},
            {"$inc": {"age": 1}}  # age + 1
        )
        print(f"$inc - 매칭: {result.matched_count}, 수정: {result.modified_count}")

        # $push: 배열에 요소 추가
        result = collection.update_one(
            {"name": "홍길동"},
            {"$push": {"hobbies": "게임"}}
        )
        print(f"$push - 매칭: {result.matched_count}, 수정: {result.modified_count}")

        # $pull: 배열에서 요소 제거
        result = collection.update_one(
            {"name": "홍길동"},
            {"$pull": {"hobbies": "게임"}}
        )
        print(f"$pull - 매칭: {result.matched_count}, 수정: {result.modified_count}")

        # 수정된 문서 확인
        user = collection.find_one({"name": "홍길동"})
        print(f"수정된 문서: {user}")

    def update_many_example(self):
        """
        여러 문서 수정 예제

        update_many(): 조건에 맞는 모든 문서 수정
        """
        print("\n" + "=" * 60)
        print("6. update_many() - 여러 문서 수정")
        print("=" * 60)

        collection = self.db["users"]

        # 30세 이상 모든 사용자에게 'senior' 태그 추가
        result = collection.update_many(
            {"age": {"$gte": 30}},
            {"$set": {"category": "senior"}}
        )
        print(f"매칭: {result.matched_count}, 수정: {result.modified_count}")

        # 확인
        print("\nsenior 카테고리 사용자:")
        for user in collection.find({"category": "senior"}):
            print(f"  - {user.get('name')}: {user.get('age')}세")

    def upsert_example(self):
        """
        Upsert 예제 (Update + Insert)

        upsert=True: 문서가 없으면 새로 생성
        """
        print("\n" + "=" * 60)
        print("7. upsert - 없으면 생성, 있으면 수정")
        print("=" * 60)

        collection = self.db["users"]

        # 존재하지 않는 문서에 upsert
        result = collection.update_one(
            {"name": "신사임당"},
            {
                "$set": {
                    "email": "shin@example.com",
                    "age": 48,
                    "created_at": datetime.now()
                }
            },
            upsert=True
        )

        print(f"매칭: {result.matched_count}, 수정: {result.modified_count}")
        print(f"Upserted ID: {result.upserted_id}")

        # 확인
        user = collection.find_one({"name": "신사임당"})
        print(f"Upserted 문서: {user}")

    # =========================================================================
    # DELETE 연산
    # =========================================================================

    def delete_one_example(self):
        """
        단일 문서 삭제 예제

        delete_one(): 조건에 맞는 첫 번째 문서 삭제
        """
        print("\n" + "=" * 60)
        print("8. delete_one() - 단일 문서 삭제")
        print("=" * 60)

        collection = self.db["users"]

        # 삭제 전 확인
        user = collection.find_one({"name": "신사임당"})
        print(f"삭제 전: {user}")

        # 문서 삭제
        result = collection.delete_one({"name": "신사임당"})
        print(f"삭제된 문서 수: {result.deleted_count}")

        # 삭제 후 확인
        user = collection.find_one({"name": "신사임당"})
        print(f"삭제 후: {user}")

    def delete_many_example(self):
        """
        여러 문서 삭제 예제

        delete_many(): 조건에 맞는 모든 문서 삭제
        """
        print("\n" + "=" * 60)
        print("9. delete_many() - 여러 문서 삭제")
        print("=" * 60)

        collection = self.db["users"]

        # 삭제 전 개수
        count_before = collection.count_documents({})
        print(f"삭제 전 문서 수: {count_before}")

        # senior 카테고리 삭제
        result = collection.delete_many({"category": "senior"})
        print(f"삭제된 문서 수: {result.deleted_count}")

        # 삭제 후 개수
        count_after = collection.count_documents({})
        print(f"삭제 후 문서 수: {count_after}")

    def cleanup(self):
        """테스트 데이터 정리"""
        print("\n" + "=" * 60)
        print("10. 테스트 데이터 정리")
        print("=" * 60)

        # users 컬렉션의 모든 문서 삭제
        result = self.db["users"].delete_many({})
        print(f"삭제된 문서 수: {result.deleted_count}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MongoDB CRUD 연산 학습")
    print("=" * 60)

    # MongoDB 연결 (환경에 맞게 수정)
    crud = None
    try:
        crud = MongoDBCRUD(
            host="localhost",
            port=27017,
            # username="admin",  # 인증 필요시 주석 해제
            # password="password",
            database="study_db"
        )

        # CREATE 연산
        crud.insert_one_example()
        crud.insert_many_example()

        # READ 연산
        crud.find_one_example()
        crud.find_many_example()

        # UPDATE 연산
        crud.update_one_example()
        crud.update_many_example()
        crud.upsert_example()

        # DELETE 연산
        crud.delete_one_example()
        crud.delete_many_example()

        # 정리
        crud.cleanup()

    except ConnectionFailure:
        print("\n[TIP] MongoDB가 실행 중인지 확인하세요:")
        print("  docker run -d -p 27017:27017 --name mongodb mongo:7.0")
    finally:
        if crud:
            crud.close()


if __name__ == "__main__":
    main()
