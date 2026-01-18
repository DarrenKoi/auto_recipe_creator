"""
MongoDB 실전 예제 - 레시피 저장 시스템

이 파일에서는 auto_recipe_creator 프로젝트에 적용할 수 있는
실제 레시피 데이터 관리 시스템을 구현합니다.

기능:
- 레시피 CRUD
- 재료별/카테고리별 검색
- 조리 시간 기반 필터링
- 난이도별 통계
- 인기 레시피 추천
"""

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json


class Difficulty(Enum):
    """레시피 난이도"""
    EASY = "쉬움"
    MEDIUM = "보통"
    HARD = "어려움"


@dataclass
class Ingredient:
    """재료 데이터 클래스"""
    name: str
    amount: str
    unit: str
    optional: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recipe:
    """레시피 데이터 클래스"""
    title: str
    description: str
    category: str
    difficulty: Difficulty
    prep_time: int  # 분
    cook_time: int  # 분
    servings: int
    ingredients: List[Ingredient]
    steps: List[str]
    tags: List[str]
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    author: str = "시스템"
    views: int = 0
    likes: int = 0
    created_at: datetime = None
    updated_at: datetime = None

    def to_dict(self) -> dict:
        data = {
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty.value,
            "prep_time": self.prep_time,
            "cook_time": self.cook_time,
            "total_time": self.prep_time + self.cook_time,
            "servings": self.servings,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "steps": self.steps,
            "tags": self.tags,
            "image_url": self.image_url,
            "video_url": self.video_url,
            "author": self.author,
            "views": self.views,
            "likes": self.likes,
            "created_at": self.created_at or datetime.now(),
            "updated_at": self.updated_at or datetime.now()
        }
        return data


class RecipeManager:
    """레시피 관리 클래스"""

    def __init__(self, uri: str = "mongodb://localhost:27017/", db_name: str = "recipe_db"):
        """
        Args:
            uri: MongoDB 연결 문자열
            db_name: 데이터베이스 이름
        """
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.recipes = self.db["recipes"]
        self.users = self.db["users"]

        try:
            self.client.admin.command('ping')
            print(f"[OK] MongoDB 연결 성공: {db_name}")
            self._setup_indexes()
        except ConnectionFailure as e:
            print(f"[ERROR] 연결 실패: {e}")
            raise

    def _setup_indexes(self):
        """필수 인덱스 설정"""
        # 제목 유니크 인덱스
        self.recipes.create_index(
            [("title", ASCENDING)],
            unique=True,
            name="title_unique"
        )

        # 카테고리 + 난이도 복합 인덱스
        self.recipes.create_index(
            [("category", ASCENDING), ("difficulty", ASCENDING)],
            name="category_difficulty"
        )

        # 조리 시간 인덱스
        self.recipes.create_index(
            [("total_time", ASCENDING)],
            name="total_time"
        )

        # 태그 멀티키 인덱스
        self.recipes.create_index(
            [("tags", ASCENDING)],
            name="tags"
        )

        # 재료 이름 인덱스
        self.recipes.create_index(
            [("ingredients.name", ASCENDING)],
            name="ingredient_name"
        )

        # 텍스트 검색 인덱스
        self.recipes.create_index(
            [("title", TEXT), ("description", TEXT), ("tags", TEXT)],
            name="text_search",
            default_language="korean"
        )

        # 인기도 인덱스 (조회수 + 좋아요)
        self.recipes.create_index(
            [("views", DESCENDING), ("likes", DESCENDING)],
            name="popularity"
        )

        print("[INFO] 인덱스 설정 완료")

    # =========================================================================
    # CRUD 연산
    # =========================================================================

    def create_recipe(self, recipe: Recipe) -> str:
        """
        레시피 생성

        Args:
            recipe: 레시피 객체

        Returns:
            생성된 레시피 ID
        """
        try:
            result = self.recipes.insert_one(recipe.to_dict())
            print(f"[OK] 레시피 생성: {recipe.title}")
            return str(result.inserted_id)
        except DuplicateKeyError:
            print(f"[ERROR] 중복된 레시피 제목: {recipe.title}")
            raise

    def get_recipe(self, recipe_id: str = None, title: str = None) -> Optional[Dict]:
        """
        레시피 조회

        Args:
            recipe_id: 레시피 ID
            title: 레시피 제목

        Returns:
            레시피 문서 또는 None
        """
        from bson.objectid import ObjectId

        if recipe_id:
            query = {"_id": ObjectId(recipe_id)}
        elif title:
            query = {"title": title}
        else:
            return None

        recipe = self.recipes.find_one(query)

        # 조회수 증가
        if recipe:
            self.recipes.update_one(
                {"_id": recipe["_id"]},
                {"$inc": {"views": 1}}
            )

        return recipe

    def update_recipe(self, recipe_id: str, updates: Dict) -> bool:
        """
        레시피 수정

        Args:
            recipe_id: 레시피 ID
            updates: 수정할 필드들

        Returns:
            수정 성공 여부
        """
        from bson.objectid import ObjectId

        updates["updated_at"] = datetime.now()

        # total_time 자동 계산
        if "prep_time" in updates or "cook_time" in updates:
            recipe = self.get_recipe(recipe_id)
            prep = updates.get("prep_time", recipe.get("prep_time", 0))
            cook = updates.get("cook_time", recipe.get("cook_time", 0))
            updates["total_time"] = prep + cook

        result = self.recipes.update_one(
            {"_id": ObjectId(recipe_id)},
            {"$set": updates}
        )

        return result.modified_count > 0

    def delete_recipe(self, recipe_id: str) -> bool:
        """
        레시피 삭제

        Args:
            recipe_id: 레시피 ID

        Returns:
            삭제 성공 여부
        """
        from bson.objectid import ObjectId

        result = self.recipes.delete_one({"_id": ObjectId(recipe_id)})
        return result.deleted_count > 0

    # =========================================================================
    # 검색 기능
    # =========================================================================

    def search_recipes(
        self,
        keyword: str = None,
        category: str = None,
        difficulty: Difficulty = None,
        max_time: int = None,
        ingredients: List[str] = None,
        tags: List[str] = None,
        sort_by: str = "created_at",
        sort_order: int = -1,
        page: int = 1,
        page_size: int = 10
    ) -> Dict:
        """
        레시피 검색

        Args:
            keyword: 검색 키워드 (제목, 설명, 태그)
            category: 카테고리
            difficulty: 난이도
            max_time: 최대 조리 시간 (분)
            ingredients: 포함해야 할 재료 목록
            tags: 포함해야 할 태그 목록
            sort_by: 정렬 기준
            sort_order: 정렬 순서 (1: 오름차순, -1: 내림차순)
            page: 페이지 번호
            page_size: 페이지 크기

        Returns:
            검색 결과 딕셔너리
        """
        query = {}

        # 텍스트 검색
        if keyword:
            query["$text"] = {"$search": keyword}

        # 카테고리 필터
        if category:
            query["category"] = category

        # 난이도 필터
        if difficulty:
            query["difficulty"] = difficulty.value

        # 조리 시간 필터
        if max_time:
            query["total_time"] = {"$lte": max_time}

        # 재료 필터 (모든 재료 포함)
        if ingredients:
            query["ingredients.name"] = {"$all": ingredients}

        # 태그 필터 (모든 태그 포함)
        if tags:
            query["tags"] = {"$all": tags}

        # 총 개수
        total_count = self.recipes.count_documents(query)

        # 페이지네이션
        skip = (page - 1) * page_size

        # 쿼리 실행
        cursor = self.recipes.find(query)

        # 텍스트 검색 시 점수 정렬
        if keyword:
            cursor = cursor.sort([("score", {"$meta": "textScore"})])
        else:
            cursor = cursor.sort(sort_by, sort_order)

        cursor = cursor.skip(skip).limit(page_size)

        recipes = list(cursor)

        return {
            "recipes": recipes,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size
        }

    def search_by_ingredients(self, ingredients: List[str], match_all: bool = False) -> List[Dict]:
        """
        재료로 레시피 검색

        Args:
            ingredients: 재료 목록
            match_all: True면 모든 재료 포함, False면 하나라도 포함

        Returns:
            레시피 목록
        """
        if match_all:
            query = {"ingredients.name": {"$all": ingredients}}
        else:
            query = {"ingredients.name": {"$in": ingredients}}

        return list(self.recipes.find(query).limit(20))

    # =========================================================================
    # 통계 및 추천
    # =========================================================================

    def get_popular_recipes(self, limit: int = 10) -> List[Dict]:
        """인기 레시피 조회"""
        pipeline = [
            {
                "$addFields": {
                    "popularity_score": {
                        "$add": ["$views", {"$multiply": ["$likes", 10]}]
                    }
                }
            },
            {"$sort": {"popularity_score": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "title": 1,
                    "category": 1,
                    "difficulty": 1,
                    "total_time": 1,
                    "views": 1,
                    "likes": 1,
                    "popularity_score": 1
                }
            }
        ]

        return list(self.recipes.aggregate(pipeline))

    def get_quick_recipes(self, max_time: int = 30) -> List[Dict]:
        """빠른 레시피 조회 (30분 이내)"""
        return list(
            self.recipes.find(
                {"total_time": {"$lte": max_time}},
                {"title": 1, "category": 1, "total_time": 1, "difficulty": 1}
            ).sort("total_time", 1).limit(10)
        )

    def get_statistics(self) -> Dict:
        """레시피 통계"""
        pipeline = [
            {
                "$facet": {
                    # 전체 통계
                    "overall": [
                        {
                            "$group": {
                                "_id": None,
                                "total_recipes": {"$sum": 1},
                                "total_views": {"$sum": "$views"},
                                "total_likes": {"$sum": "$likes"},
                                "avg_time": {"$avg": "$total_time"}
                            }
                        }
                    ],
                    # 카테고리별 통계
                    "by_category": [
                        {
                            "$group": {
                                "_id": "$category",
                                "count": {"$sum": 1},
                                "avg_time": {"$avg": "$total_time"}
                            }
                        },
                        {"$sort": {"count": -1}}
                    ],
                    # 난이도별 통계
                    "by_difficulty": [
                        {
                            "$group": {
                                "_id": "$difficulty",
                                "count": {"$sum": 1}
                            }
                        }
                    ]
                }
            }
        ]

        result = list(self.recipes.aggregate(pipeline))[0]

        return {
            "overall": result["overall"][0] if result["overall"] else {},
            "by_category": result["by_category"],
            "by_difficulty": result["by_difficulty"]
        }

    def get_category_list(self) -> List[str]:
        """카테고리 목록"""
        return self.recipes.distinct("category")

    def get_tag_list(self) -> List[str]:
        """태그 목록"""
        return self.recipes.distinct("tags")

    # =========================================================================
    # 좋아요 기능
    # =========================================================================

    def like_recipe(self, recipe_id: str) -> bool:
        """레시피 좋아요"""
        from bson.objectid import ObjectId

        result = self.recipes.update_one(
            {"_id": ObjectId(recipe_id)},
            {"$inc": {"likes": 1}}
        )
        return result.modified_count > 0

    def unlike_recipe(self, recipe_id: str) -> bool:
        """레시피 좋아요 취소"""
        from bson.objectid import ObjectId

        result = self.recipes.update_one(
            {"_id": ObjectId(recipe_id)},
            {"$inc": {"likes": -1}}
        )
        return result.modified_count > 0

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def seed_sample_data(self):
        """샘플 데이터 삽입"""
        print("\n샘플 레시피 데이터 생성 중...")

        recipes = [
            Recipe(
                title="김치찌개",
                description="돼지고기와 김치로 만드는 얼큰한 찌개",
                category="찌개/국물",
                difficulty=Difficulty.EASY,
                prep_time=10,
                cook_time=20,
                servings=2,
                ingredients=[
                    Ingredient("김치", "200", "g"),
                    Ingredient("돼지고기", "150", "g"),
                    Ingredient("두부", "1/2", "모"),
                    Ingredient("대파", "1", "대"),
                    Ingredient("고춧가루", "1", "큰술"),
                ],
                steps=[
                    "돼지고기를 한입 크기로 썬다",
                    "냄비에 돼지고기를 볶는다",
                    "김치를 넣고 함께 볶는다",
                    "물을 붓고 끓인다",
                    "두부와 대파를 넣고 5분 더 끓인다"
                ],
                tags=["한식", "찌개", "얼큰", "돼지고기", "김치"],
                author="요리왕"
            ),
            Recipe(
                title="계란볶음밥",
                description="간단하고 맛있는 계란볶음밥",
                category="볶음밥",
                difficulty=Difficulty.EASY,
                prep_time=5,
                cook_time=10,
                servings=1,
                ingredients=[
                    Ingredient("밥", "1", "공기"),
                    Ingredient("계란", "2", "개"),
                    Ingredient("대파", "1/2", "대"),
                    Ingredient("간장", "1", "큰술"),
                    Ingredient("참기름", "1", "작은술"),
                ],
                steps=[
                    "계란을 풀어 스크램블을 만든다",
                    "대파를 송송 썬다",
                    "팬에 밥을 넣고 볶는다",
                    "스크램블과 대파를 넣는다",
                    "간장과 참기름으로 간한다"
                ],
                tags=["한식", "볶음밥", "간단", "한그릇"],
                author="집밥러"
            ),
            Recipe(
                title="된장찌개",
                description="구수한 된장찌개",
                category="찌개/국물",
                difficulty=Difficulty.EASY,
                prep_time=15,
                cook_time=20,
                servings=2,
                ingredients=[
                    Ingredient("된장", "2", "큰술"),
                    Ingredient("두부", "1/2", "모"),
                    Ingredient("감자", "1", "개"),
                    Ingredient("호박", "1/4", "개"),
                    Ingredient("양파", "1/2", "개"),
                    Ingredient("청양고추", "1", "개", optional=True),
                ],
                steps=[
                    "감자와 호박을 깍둑썰기한다",
                    "양파를 얇게 썬다",
                    "냄비에 물을 붓고 된장을 푼다",
                    "감자를 넣고 끓인다",
                    "호박, 양파, 두부를 넣고 끓인다"
                ],
                tags=["한식", "찌개", "된장", "건강"],
                author="요리왕"
            ),
            Recipe(
                title="파스타 알리오 올리오",
                description="마늘과 올리브오일의 심플한 파스타",
                category="면요리",
                difficulty=Difficulty.MEDIUM,
                prep_time=10,
                cook_time=15,
                servings=1,
                ingredients=[
                    Ingredient("스파게티면", "100", "g"),
                    Ingredient("마늘", "5", "쪽"),
                    Ingredient("올리브오일", "4", "큰술"),
                    Ingredient("페페론치노", "2", "개"),
                    Ingredient("파슬리", "약간", ""),
                ],
                steps=[
                    "마늘을 얇게 슬라이스한다",
                    "끓는 물에 소금을 넣고 면을 삶는다",
                    "팬에 올리브오일과 마늘을 넣고 약불에 볶는다",
                    "면수 2큰술을 넣고 유화시킨다",
                    "삶은 면을 넣고 버무린다"
                ],
                tags=["양식", "파스타", "마늘", "심플"],
                author="파스타장인"
            ),
            Recipe(
                title="소고기 미역국",
                description="생일이나 특별한 날 먹는 소고기 미역국",
                category="찌개/국물",
                difficulty=Difficulty.MEDIUM,
                prep_time=30,
                cook_time=40,
                servings=4,
                ingredients=[
                    Ingredient("건미역", "20", "g"),
                    Ingredient("소고기", "150", "g"),
                    Ingredient("참기름", "2", "큰술"),
                    Ingredient("국간장", "2", "큰술"),
                    Ingredient("다진마늘", "1", "큰술"),
                ],
                steps=[
                    "미역을 물에 불린다",
                    "소고기를 먹기 좋게 썬다",
                    "냄비에 참기름을 두르고 소고기를 볶는다",
                    "미역을 넣고 함께 볶는다",
                    "물을 붓고 40분간 끓인다"
                ],
                tags=["한식", "국", "소고기", "미역", "생일"],
                author="어머니"
            )
        ]

        for recipe in recipes:
            try:
                self.create_recipe(recipe)
            except DuplicateKeyError:
                print(f"  [SKIP] 이미 존재: {recipe.title}")

        print(f"샘플 데이터 생성 완료: {len(recipes)}개")

    def cleanup(self):
        """데이터 정리"""
        self.recipes.delete_many({})
        print("[INFO] 레시피 데이터 정리 완료")

    def close(self):
        """연결 종료"""
        self.client.close()
        print("[INFO] MongoDB 연결 종료")


def main():
    """메인 함수"""
    print("=" * 60)
    print("MongoDB 실전 예제 - 레시피 저장 시스템")
    print("=" * 60)

    manager = None
    try:
        manager = RecipeManager()

        # 샘플 데이터 생성
        manager.seed_sample_data()

        # 테스트
        print("\n" + "=" * 60)
        print("레시피 검색 테스트")
        print("=" * 60)

        # 1. 전체 검색
        print("\n1. 전체 레시피:")
        result = manager.search_recipes()
        for r in result["recipes"]:
            print(f"  - {r['title']} ({r['category']}, {r['difficulty']})")

        # 2. 키워드 검색
        print("\n2. '찌개' 키워드 검색:")
        result = manager.search_recipes(keyword="찌개")
        for r in result["recipes"]:
            print(f"  - {r['title']}")

        # 3. 카테고리 + 난이도 필터
        print("\n3. 찌개/국물 + 쉬움:")
        result = manager.search_recipes(category="찌개/국물", difficulty=Difficulty.EASY)
        for r in result["recipes"]:
            print(f"  - {r['title']}")

        # 4. 빠른 레시피 (20분 이내)
        print("\n4. 20분 이내 레시피:")
        quick = manager.get_quick_recipes(max_time=20)
        for r in quick:
            print(f"  - {r['title']}: {r['total_time']}분")

        # 5. 재료로 검색
        print("\n5. '두부'가 포함된 레시피:")
        result = manager.search_by_ingredients(["두부"])
        for r in result:
            print(f"  - {r['title']}")

        # 6. 통계
        print("\n6. 레시피 통계:")
        stats = manager.get_statistics()
        if stats["overall"]:
            print(f"  총 레시피: {stats['overall'].get('total_recipes', 0)}개")
            print(f"  평균 조리시간: {stats['overall'].get('avg_time', 0):.0f}분")

        print("\n  카테고리별:")
        for cat in stats["by_category"]:
            print(f"    - {cat['_id']}: {cat['count']}개")

        # 7. 카테고리/태그 목록
        print("\n7. 카테고리 목록:", manager.get_category_list())
        print("   태그 목록:", manager.get_tag_list()[:5], "...")

        # 정리 (선택사항)
        # manager.cleanup()

    except ConnectionFailure:
        print("\n[TIP] MongoDB 서버가 실행 중인지 확인하세요:")
        print("  docker run -d -p 27017:27017 --name mongodb mongo:7.0")
    finally:
        if manager:
            manager.close()


if __name__ == "__main__":
    main()
