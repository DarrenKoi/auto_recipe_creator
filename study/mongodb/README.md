# MongoDB 학습 가이드

## 목차
1. [MongoDB란?](#mongodb란)
2. [설치 및 설정](#설치-및-설정)
3. [핵심 개념](#핵심-개념)
4. [Python 연동](#python-연동)
5. [학습 파일 구성](#학습-파일-구성)

---

## MongoDB란?

MongoDB는 **문서 지향(Document-Oriented) NoSQL 데이터베이스**입니다.

### 특징
- **스키마리스(Schemaless)**: 유연한 데이터 구조
- **JSON/BSON 형식**: 문서 기반 데이터 저장
- **수평적 확장(Sharding)**: 대용량 데이터 처리
- **고성능**: 인덱싱, 복제, 샤딩 지원
- **풍부한 쿼리**: 집계 파이프라인, 텍스트 검색

### RDBMS vs MongoDB 용어 비교

| RDBMS | MongoDB | 설명 |
|-------|---------|------|
| Database | Database | 데이터베이스 |
| Table | Collection | 테이블/컬렉션 |
| Row | Document | 행/문서 |
| Column | Field | 열/필드 |
| Primary Key | _id | 기본 키 |
| JOIN | $lookup | 조인 연산 |

---

## 설치 및 설정

### 1. MongoDB 서버 설치

#### Docker (권장)
```bash
# MongoDB 컨테이너 실행
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  -v mongodb_data:/data/db \
  mongo:7.0

# 컨테이너 접속
docker exec -it mongodb mongosh -u admin -p password
```

#### Ubuntu/Debian
```bash
# GPG 키 추가
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | \
  sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# 저장소 추가
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] \
  https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# 설치
sudo apt-get update
sudo apt-get install -y mongodb-org

# 서비스 시작
sudo systemctl start mongod
sudo systemctl enable mongod
```

### 2. Python 드라이버 설치
```bash
pip install pymongo
# 또는
pip install -r requirements.txt
```

---

## 핵심 개념

### 1. 문서(Document)
MongoDB의 기본 데이터 단위입니다. JSON과 유사한 BSON 형식으로 저장됩니다.

```json
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "name": "김철수",
  "age": 30,
  "email": "kim@example.com",
  "address": {
    "city": "서울",
    "zipcode": "12345"
  },
  "hobbies": ["독서", "게임", "운동"],
  "created_at": ISODate("2024-01-15T09:00:00Z")
}
```

### 2. 컬렉션(Collection)
문서들의 그룹입니다. RDBMS의 테이블과 유사하지만 스키마가 고정되지 않습니다.

### 3. 데이터베이스(Database)
컬렉션들의 물리적 컨테이너입니다.

### 4. _id 필드
모든 문서는 고유한 `_id` 필드를 가집니다. 지정하지 않으면 자동으로 ObjectId가 생성됩니다.

---

## Python 연동

### 기본 연결
```python
from pymongo import MongoClient

# 연결 문자열
uri = "mongodb://admin:password@localhost:27017/"

# 클라이언트 생성
client = MongoClient(uri)

# 데이터베이스 선택
db = client["mydb"]

# 컬렉션 선택
collection = db["users"]
```

### CRUD 연산 요약

```python
# Create
collection.insert_one({"name": "홍길동", "age": 25})
collection.insert_many([{"name": "이순신"}, {"name": "강감찬"}])

# Read
collection.find_one({"name": "홍길동"})
collection.find({"age": {"$gte": 20}})

# Update
collection.update_one({"name": "홍길동"}, {"$set": {"age": 26}})
collection.update_many({"age": {"$lt": 30}}, {"$inc": {"age": 1}})

# Delete
collection.delete_one({"name": "홍길동"})
collection.delete_many({"age": {"$lt": 20}})
```

---

## 학습 파일 구성

| 파일 | 내용 | 난이도 |
|------|------|--------|
| `01_basic_crud.py` | 기본 CRUD 연산 | ⭐ |
| `02_queries.py` | 다양한 쿼리 방법 | ⭐⭐ |
| `03_aggregation.py` | 집계 파이프라인 | ⭐⭐⭐ |
| `04_indexing.py` | 인덱스 생성 및 활용 | ⭐⭐ |
| `05_practical_example.py` | 실전 예제 (레시피 저장) | ⭐⭐⭐ |

### 실행 방법
```bash
# 의존성 설치
pip install -r requirements.txt

# 개별 파일 실행
python 01_basic_crud.py
python 02_queries.py
python 03_aggregation.py
python 04_indexing.py
python 05_practical_example.py
```

---

## 유용한 명령어 (mongosh)

```javascript
// 데이터베이스 목록
show dbs

// 데이터베이스 선택/생성
use mydb

// 컬렉션 목록
show collections

// 컬렉션 생성
db.createCollection("users")

// 문서 개수
db.users.countDocuments()

// 컬렉션 삭제
db.users.drop()

// 데이터베이스 삭제
db.dropDatabase()
```

---

## 참고 자료

- [MongoDB 공식 문서](https://www.mongodb.com/docs/)
- [PyMongo 문서](https://pymongo.readthedocs.io/)
- [MongoDB University](https://learn.mongodb.com/)
